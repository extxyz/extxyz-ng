/* native read parsed using `nom`
*/
use crate::error::ExtxyzError;
use extxyz_types::{DictHandler, FloatNum, Frame, Text, Value};
use nom::{
    self,
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::{
        complete::{self, multispace0, space0, space1},
        streaming,
    },
    combinator::{all_consuming, map, map_res, not, opt, peek, recognize, verify},
    multi::{many0, separated_list0, separated_list1},
    number,
    sequence::{delimited, separated_pair, terminated},
    IResult, Parser,
};
use std::{
    collections::BTreeMap,
    io::{self, BufRead},
};

/// read from a buf reader and return an `FrameReader` which is an interator.
pub fn read_frames<'a, R>(rd: &'a mut R) -> FrameReader<'a, R>
where
    R: BufRead,
{
    FrameReader {
        rd,
        finished: false,
    }
}

pub struct FrameReader<'a, R> {
    // None as done marker
    rd: &'a mut R,
    finished: bool,
}

impl<'a, R> Iterator for FrameReader<'a, R>
where
    R: BufRead,
{
    type Item = Result<Frame, ExtxyzError>;

    fn next(&mut self) -> Option<Self::Item> {
        // fast finished
        if self.finished {
            return None;
        }

        match _read_frame_native_new(self.rd, None) {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(err) => Some(Err(ExtxyzError::Io(err))),
        }
    }
}

pub fn read_frame<R>(rd: &mut R) -> Result<Frame, ExtxyzError>
where
    R: BufRead,
{
    let Some(frame) = _read_frame_native_new(rd, None)? else {
        return Err(ExtxyzError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "does not parse anything from reader",
        )));
    };
    Ok(frame)
}

pub(crate) fn _read_frame_native_new<R>(
    rd: &mut R,
    comment_override: Option<&str>,
) -> io::Result<Option<Frame>>
// XXX: still IResult if it is better
where
    R: BufRead,
{
    let mut maybe_natoms_line = String::new();
    rd.read_line(&mut maybe_natoms_line)?;
    if maybe_natoms_line.is_empty() {
        return Ok(None);
    }

    // parse number of lines
    let natoms_line_as_bytes = maybe_natoms_line.as_bytes();
    let (_, natoms) = parse_natoms(natoms_line_as_bytes).map_err(|e| {
        let es = match e {
            nom::Err::Incomplete(_) => "nom incomplete streaming".to_string(),
            nom::Err::Error(err) | nom::Err::Failure(err) => {
                format!(
                    "{:?}: {}",
                    err.code,
                    str::from_utf8(err.input).unwrap_or("unrecognized u8 input")
                )
            }
        };
        io::Error::new(io::ErrorKind::InvalidData, es)
    })?;

    let mut maybe_info_line = String::new();
    rd.read_line(&mut maybe_info_line)?;
    if maybe_info_line.is_empty() {
        return Ok(None);
    }

    let info_line_as_bytes = maybe_info_line.as_bytes();
    let (_, (info, prop_shape)) = parse_info(info_line_as_bytes).map_err(|e| {
        let es = match e {
            nom::Err::Incomplete(_) => "nom incomplete streaming".to_string(),
            nom::Err::Error(err) | nom::Err::Failure(err) => {
                format!(
                    "{:?}: {}",
                    err.code,
                    str::from_utf8(err.input).unwrap_or("unrecognized u8 input")
                )
            }
        };
        io::Error::new(io::ErrorKind::InvalidData, es)
    })?;

    // init the arrs from the shape, in order to avoid innermiddle allocation
    let mut arrs: Vec<(String, Value)> = prop_shape
        .iter()
        .map(|(name, ty, n)| {
            let value = match (ty, n) {
                (Ty::I, 1) => Value::VecInteger(Vec::with_capacity(natoms), natoms as u32),
                (Ty::R, 1) => Value::VecFloat(Vec::with_capacity(natoms), natoms as u32),
                (Ty::L, 1) => Value::VecBool(Vec::with_capacity(natoms), natoms as u32),
                (Ty::S, 1) => Value::VecText(Vec::with_capacity(natoms), natoms as u32),

                (Ty::I, nc) => {
                    Value::MatrixInteger(Vec::with_capacity(natoms), (natoms as u32, *nc as u32))
                }
                (Ty::R, nc) => {
                    Value::MatrixFloat(Vec::with_capacity(natoms), (natoms as u32, *nc as u32))
                }
                (Ty::L, nc) => {
                    Value::MatrixBool(Vec::with_capacity(natoms), (natoms as u32, *nc as u32))
                }
                (Ty::S, nc) => {
                    Value::MatrixText(Vec::with_capacity(natoms), (natoms as u32, *nc as u32))
                }
            };

            (name.to_string(), value)
        })
        .collect();

    let mut natoms_to_read = natoms;
    // TODO: validate natoms and number of rows of the arr
    loop {
        let buf = rd.fill_buf()?;
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF reached before parsing frame",
            ));
        }

        match parse_xyz_by_lines(buf, natoms_to_read, &prop_shape, &mut arrs) {
            Ok((remain, nat)) => {
                let len_read = buf.len() - remain.len();
                rd.consume(len_read);

                natoms_to_read -= nat;
                if natoms_to_read == 0 {
                    break;
                } else if natoms_to_read > 0 {
                    continue;
                } else {
                    // < 0
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "too many atoms than expected",
                    ));
                }
            }
            Err(e) => {
                let es = match e {
                    nom::Err::Incomplete(_) => "nom incomplete streaming".to_string(),
                    nom::Err::Error(err) | nom::Err::Failure(err) => {
                        format!(
                            "{:?}: {}",
                            err.code,
                            str::from_utf8(err.input).unwrap_or("unrecognized u8 input")
                        )
                    }
                };
                return Err(io::Error::new(io::ErrorKind::InvalidData, es));
            }
        }
    }

    let mut frame = Frame {
        natoms: natoms as u32,
        info: DictHandler(info),
        arrs: DictHandler(arrs),
    };

    // // any remain spaces
    // let (input, _) = multispace0::<_, nom::error::Error<&[u8]>>(input)
    //     .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    if let Some(comment) = comment_override {
        frame.set_comment(comment);
    }

    Ok(Some(frame))
}

// XXX: can I lru cache this call? this can be useful when parsing frames because the same info
// lines will be kept on parsed again and again.
fn key_value(inp: &[u8]) -> IResult<&[u8], (&[u8], &[u8])> {
    let (inp, (k, v)) = separated_pair(
        delimited(
            multispace0,
            verify(
                take_while1(|c: u8| c != b'=' && !c.is_ascii_whitespace()),
                |s: &[u8]| recognize_kv_left(s).is_ok(),
            ),
            multispace0,
        ),
        tag(&b"="[..]),
        delimited(multispace0, recognize_kv_right, multispace0),
    )
    .parse(inp)?;
    Ok((inp, (k, v)))
}

fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn recognize_int(inp: &[u8]) -> IResult<&[u8], &[u8]> {
    terminated(
        recognize(complete::i32),
        peek(not(take_while1(is_ident_char))),
    )
    .parse(inp)
}

// i32
fn parse_int(inp: &[u8]) -> IResult<&[u8], Value> {
    map_res(recognize_int, |bytes: &[u8]| {
        let s = std::str::from_utf8(bytes)
            .map_err(|_| nom::error::Error::new(bytes, nom::error::ErrorKind::Char))?;
        let i = s
            .parse::<i32>()
            .map_err(|_| nom::error::Error::new(bytes, nom::error::ErrorKind::Digit))?;
        Ok::<Value, nom::error::Error<&[u8]>>(Value::Integer(i.into()))
    })
    .parse(inp)
}

fn recognize_float(inp: &[u8]) -> IResult<&[u8], &[u8]> {
    // number::complete::double will parse an integer into a float, this is what I don't want
    // I parse twice here, using recognize_float_parts to get the fraction part and error out if it
    // is a pure integer.
    // More performant one is reimplement number::complete::double, but I dont bother do it.
    let (inp_, (_, _, fraction, _)) = number::complete::recognize_float_parts(inp)?;
    if fraction.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            inp_,
            nom::error::ErrorKind::Float,
        )));
    }
    let len = inp.len() - inp_.len();
    Ok((inp_, &inp[..len]))
}

fn parse_float(inp: &[u8]) -> IResult<&[u8], Value> {
    let (remain, inp) = recognize_float.parse(inp)?;
    let (_, float) = number::complete::double
        .map(|i| Value::Float(i.into()))
        .parse(inp)?;

    Ok((remain, float))
}

fn parse_bool(inp: &[u8]) -> IResult<&[u8], Value> {
    // T or F or [tT]rue or [fF]alse or TRUE or FALSE
    alt((
        tag("true").map(|_| Value::Bool(true.into())),
        tag("false").map(|_| Value::Bool(false.into())),
        tag("True").map(|_| Value::Bool(true.into())),
        tag("False").map(|_| Value::Bool(false.into())),
        tag("TRUE").map(|_| Value::Bool(true.into())),
        tag("FALSE").map(|_| Value::Bool(false.into())),
        tag("T").map(|_| Value::Bool(true.into())),
        tag("F").map(|_| Value::Bool(false.into())),
    ))
    .parse(inp)
}

fn recognize_bool(inp: &[u8]) -> IResult<&[u8], &[u8]> {
    // XXX: should to v.v
    recognize(parse_bool).parse(inp)
}

fn parse_bare_str(inp: &[u8]) -> IResult<&[u8], Value> {
    let (remain, inp) = recognize_bare_str.parse(inp)?;
    let s = String::from_utf8(inp.to_vec()).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
    })?;
    Ok((remain, Value::Str(Text::from(s))))
}

fn recognize_bare_str(inp: &[u8]) -> IResult<&[u8], &[u8]> {
    let (linp, s) = take_while1(|c: u8| is_ident_char(c)).parse(inp)?;
    if !s[0].is_ascii_alphanumeric() && s[0] != b'_' {
        return Err(nom::Err::Error(nom::error::Error::new(
            linp,
            nom::error::ErrorKind::Verify,
        )));
    }
    let len = inp.len() - linp.len();
    Ok((linp, &inp[..len]))
}

fn parse_quote_str(inp: &[u8]) -> IResult<&[u8], Value> {
    let parse_inner = map_res(
        many0(alt((
            take_while1(|b| b != b'\\' && b != b'"'),
            map(tag(r#"\""#), |_| b"\"".as_ref()),
            map(tag(r#"\\"#), |_| b"\\".as_ref()),
            map(tag(r#"\n"#), |_| b"\n".as_ref()),
        ))),
        |chunks: Vec<&[u8]>| {
            let s = chunks.concat();
            String::from_utf8(s).map(|s| Value::Str(Text::from(s)))
        },
    );

    let (inp, xx) = delimited(tag(b"\"".as_ref()), parse_inner, tag(b"\"".as_ref())).parse(inp)?;
    Ok((inp, xx))
}

fn parse_bare_properties_str(inp: &[u8]) -> IResult<&[u8], Value> {
    let (remain, inp) = take_while1(|c: u8| {
        c.is_ascii_alphanumeric() || c == b'_' || c == b':' || c == b'@' || c == b'/'
    })
    .parse(inp)?;
    let s = String::from_utf8(inp.to_vec()).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
    })?;
    let v = Value::Str(Text::from(s));
    Ok((remain, v))
}

fn parse_kv_right(inp: &[u8]) -> IResult<&[u8], Value> {
    // order conform with the spec, see README for spec definition.
    alt((
        parse_2d_array,
        // float before int, to avoid 3.14 -> 3
        parse_float,
        parse_int,
        // bool comes before str, to avoid boll true -> str "true"
        parse_bool,
        // quete_str first because it is most picky to parse, must have "" around
        parse_quote_str,
        // quotet_properties_str is included in parse_quote_str
        parse_bare_properties_str,
        parse_bare_str,
    ))
    .parse(inp)
}

// left part of kv, i.e. the key part, which need to be a parsable string.
fn parse_kv_left(inp: &[u8]) -> IResult<&[u8], Value> {
    alt((parse_bare_str, parse_quote_str)).parse(inp)
}

fn recognize_kv_right(inp: &[u8]) -> IResult<&[u8], &[u8]> {
    recognize(parse_kv_right).parse(inp)
}

fn recognize_kv_left(inp: &[u8]) -> IResult<&[u8], &[u8]> {
    recognize(parse_kv_left).parse(inp)
}

fn parse_2d_arr_3x3_flatten(inp: &[u8]) -> IResult<&[u8], Value> {
    let (inp, mut vals) = separated_list0(space1, parse_kv_right).parse(inp)?;
    if vals.len() != 9 {
        return Err(nom::Err::Failure(nom::error::Error::new(
            inp,
            nom::error::ErrorKind::Verify,
        )));
    }
    promote_values_1d(&mut vals).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
    })?;

    match &vals[0] {
        Value::Integer(_) => {
            let vals = vals
                .into_iter()
                .map(|v| v.as_integer().expect("not an integer"))
                .collect::<Vec<_>>();
            let row1 = vec![vals[0], vals[3], vals[6]];
            let row2 = vec![vals[1], vals[4], vals[7]];
            let row3 = vec![vals[2], vals[5], vals[8]];
            let vs = vec![row1, row2, row3];
            Ok((inp, Value::MatrixInteger(vs, (3, 3))))
        }
        Value::Float(_) => {
            let vals = vals
                .into_iter()
                .map(|v| v.as_float().expect("not a float"))
                .collect::<Vec<_>>();
            let row1 = vec![vals[0], vals[3], vals[6]];
            let row2 = vec![vals[1], vals[4], vals[7]];
            let row3 = vec![vals[2], vals[5], vals[8]];
            let vs = vec![row1, row2, row3];
            Ok((inp, Value::MatrixFloat(vs, (3, 3))))
        }
        _ => Err(nom::Err::Failure(nom::error::Error::new(
            inp,
            nom::error::ErrorKind::Verify,
        ))),
    }
}

fn parse_2d_array(inp: &[u8]) -> IResult<&[u8], Value> {
    let (inp_, vals) = delimited(
        tag(b"[".as_ref()),
        separated_list0(
            tag(b",".as_ref()),
            delimited(multispace0, parse_1d_array, multispace0),
        ),
        tag(b"]".as_ref()),
    )
    .parse(inp)?;

    debug_assert!(!vals.is_empty());

    match &vals[0] {
        Value::VecInteger(_, nc) => {
            let nc = *nc;
            let nr = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::VecInteger(i, x) = v else {
                        unreachable!()
                    };
                    debug_assert_eq!(x, nc);
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::MatrixInteger(vs, (nr as u32, nc))))
        }
        Value::VecFloat(_, nc) => {
            let nc = *nc;
            let nr = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::VecFloat(i, x) = v else {
                        unreachable!()
                    };
                    debug_assert_eq!(x, nc);
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::MatrixFloat(vs, (nr as u32, nc))))
        }
        Value::VecBool(_, nc) => {
            let nc = *nc;
            let nr = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::VecBool(i, x) = v else {
                        unreachable!()
                    };
                    debug_assert_eq!(x, nc);
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::MatrixBool(vs, (nr as u32, nc))))
        }
        Value::VecText(_, nc) => {
            let nc = *nc;
            let nr = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::VecText(i, x) = v else {
                        unreachable!()
                    };
                    debug_assert_eq!(x, nc);
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::MatrixText(vs, (nr as u32, nc))))
        }
        _ => unreachable!(),
    }
}

// fn recognize_2d_array(inp: &[u8]) -> IResult<&[u8], &[u8]> {
//     recognize(parse_2d_array).parse(inp)
// }

fn parse_1d_array(inp: &[u8]) -> IResult<&[u8], Value> {
    let (inp_, mut vals) = delimited(
        tag(b"[".as_ref()),
        separated_list0(
            tag(b",".as_ref()),
            delimited(multispace0, parse_kv_right, multispace0),
        ),
        tag(b"]".as_ref()),
    )
    .parse(inp)?;

    debug_assert!(!vals.is_empty());

    // promote by single rule:
    // only int -> float when mixed, all other mixture will fail.
    promote_values_1d(&mut vals).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(inp_, nom::error::ErrorKind::Verify))
    })?;

    match &vals[0] {
        Value::Integer(_) => {
            let n = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::Integer(i) = v else { unreachable!() };
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::VecInteger(vs, n as u32)))
        }
        Value::Float(_) => {
            let n = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::Float(i) = v else { unreachable!() };
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::VecFloat(vs, n as u32)))
        }
        Value::Bool(_) => {
            let n = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::Bool(i) = v else { unreachable!() };
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::VecBool(vs, n as u32)))
        }
        Value::Str(_) => {
            let n = vals.len();
            let vs = vals
                .into_iter()
                .map(|v| {
                    let Value::Str(i) = v else { unreachable!() };
                    i
                })
                .collect::<Vec<_>>();
            Ok((inp_, Value::VecText(vs, n as u32)))
        }
        // safe unreachable: because all branches are ruled out in promote_values_1d call.
        _ => unreachable!(),
    }
}

// fn recognize_1d_array(inp: &[u8]) -> IResult<&[u8], &[u8]> {
//     recognize(parse_1d_array).parse(inp)
// }

#[derive(Debug)]
struct InnerParseError;

impl std::fmt::Display for InnerParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inner parse error")
    }
}

impl std::error::Error for InnerParseError {}

fn promote_values_1d(vals: &mut [Value]) -> Result<(), InnerParseError> {
    if vals.is_empty() {
        return Ok(());
    }

    if vals.iter().any(|v| {
        matches!(
            v,
            Value::VecBool(_, _)
                | Value::VecText(_, _)
                | Value::VecFloat(_, _)
                | Value::VecInteger(_, _)
                | Value::MatrixBool(_, _)
                | Value::MatrixText(_, _)
                | Value::MatrixFloat(_, _)
                | Value::MatrixInteger(_, _)
                | Value::Unsupported
        )
    }) {
        return Err(InnerParseError);
    }

    let has_bool = vals.iter().any(|v| matches!(v, Value::Bool(_)));
    let has_float = vals.iter().any(|v| matches!(v, Value::Float(_)));
    let has_str = vals.iter().any(|v| matches!(v, Value::Str(_)));
    let has_int = vals.iter().any(|v| matches!(v, Value::Integer(_)));

    match (has_int, has_float, has_bool, has_str) {
        // homogeneous types in array, no promotion needed
        (true, false, false, false)
        | (false, true, false, false)
        | (false, false, true, false)
        | (false, false, false, true) => Ok(()),
        // int and float in array, promote all int to float
        (true, true, false, false) => {
            vals.iter_mut().for_each(|v| {
                if let Value::Integer(i) = v {
                    *v = Value::Float(FloatNum::from(f64::from(**i)));
                }
            });
            Ok(())
        }
        // error out if more mixture types
        (true, true, true, true)
        | (true, true, true, false)
        | (true, true, false, true)
        | (true, false, true, true)
        | (true, false, true, false)
        | (true, false, false, true)
        | (false, true, true, true)
        | (false, true, true, false)
        | (false, true, false, true)
        | (false, false, true, true) => Err(InnerParseError),
        (false, false, false, false) => unreachable!(),
    }
}

#[allow(clippy::type_complexity)]
fn parse_info_line(inp: &[u8]) -> IResult<&[u8], Vec<(&[u8], &[u8])>> {
    let (inp, kv) = delimited(
        multispace0,
        all_consuming(separated_list1(space0, key_value)),
        multispace0,
    )
    .parse(inp)?;
    Ok((inp, kv))
}

#[allow(clippy::type_complexity)]
fn parse_no_equal_sign_line(inp: &[u8]) -> IResult<&[u8], Vec<(&[u8], &[u8])>> {
    let (inp, ln) = take_while1(|c: u8| c != b'=').parse(inp)?;
    Ok((inp, vec![(&b"comment"[..], ln)]))
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum Ty {
    // integer
    I,
    // Real
    R,
    // Logic
    L,
    // String
    S,
}

type PropShape<'a> = Vec<(&'a str, Ty, u8)>;

fn parse_properties<'a>(inp: &'a [u8]) -> IResult<&'a [u8], PropShape<'a>> {
    // into triple elements chunk
    let (inp_, segments) =
        separated_list1(tag(b":".as_ref()), take_while1(|c: u8| c != b':')).parse(inp)?;

    if segments.len() % 3 != 0 {
        // TODO: verbose context error
        return Err(nom::Err::Failure(nom::error::Error::new(
            inp,
            nom::error::ErrorKind::Verify,
        )));
    }

    // TODO: check key name should not duplicate, because that is the name as keys for arrs
    let mut mp = Vec::new();
    for chunk in segments.chunks(3) {
        let id = chunk[0];
        let ty = match chunk[1] {
            b"I" => Ty::I,
            b"R" => Ty::R,
            b"L" => Ty::L,
            b"S" => Ty::S,
            _ => {
                // TODO: verbose context error
                return Err(nom::Err::Failure(nom::error::Error::new(
                    inp,
                    nom::error::ErrorKind::Verify,
                )));
            }
        };
        let nc = str::from_utf8(chunk[2])
            .map_err(|_| {
                nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
            })?
            .parse::<u8>()
            .map_err(|_| {
                nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
            })?;

        let id = str::from_utf8(id).unwrap();
        mp.push((id, ty, nc));
    }
    Ok((inp_, mp))
}

type TypInfo = Vec<(String, Value)>;
type TypPropShape<'a> = Vec<(&'a str, Ty, u8)>;

fn parse_info<'a>(input: &'a [u8]) -> IResult<&'a [u8], (TypInfo, TypPropShape<'a>)> {
    let (input, line) = terminated(
        nom::bytes::complete::take_until(&b"\n"[..]),
        complete::newline,
    )
    .parse(input)?;

    let (_, info_kv) = alt((
        all_consuming(parse_info_line),
        all_consuming(parse_no_equal_sign_line),
    ))
    .parse(line)?;

    // use BTreeMap so the info is stored in order
    let mut kv = BTreeMap::new();

    for (k, v) in info_kv {
        let old_val = kv.insert(k, v);
        // fatal when key duplicate in the info line
        // TODO: verbose context error
        if old_val.is_some() {
            return Err(nom::Err::Failure(nom::error::Error::new(
                k,
                nom::error::ErrorKind::Verify,
            )));
        }
    }

    // TODO: check "properties"/"property"/"Property" and raise error with help message.
    // TODO: check "lattice" and raise error with help message.
    //
    // The default (fallback) shape is: Properties=species:S:1:pos:R:3
    let prop_shape = kv
        .remove("Properties".as_bytes())
        .unwrap_or(b"species:S:1:pos:R:3");

    let utf8_str = str::from_utf8(prop_shape).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;
    let prop_shape_value = Value::Str(Text::from(utf8_str));
    let (_, prop_shape) = parse_properties(prop_shape)?;

    let maybe_latt = kv.remove("Lattice".as_bytes());

    // XXX: comment, latt and prop_shape better to be stored separatly from pure_kv
    let mut info = Vec::with_capacity(kv.len() + 2);
    for (k, v) in kv {
        if k == &b"comment"[..] {
            let utf8_str = str::from_utf8(v).map_err(|_| {
                nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Verify))
            })?;
            info.push(("comment".to_string(), Value::Str(Text::from(utf8_str))));
        } else {
            let (_, v) = parse_kv_right(v)?;
            info.push((String::from_utf8(k.to_vec()).expect("utf8"), v));
        }
    }

    // TODO: latt can be a matrix wrapped inside a pair of double quotes, test it
    if let Some(latt) = maybe_latt {
        let opt_quote_parse_2d_array = delimited(
            opt(tag(b"\"".as_ref())),
            parse_2d_array,
            opt(tag(b"\"".as_ref())),
        );
        let opt_quote_parse_2d_arr_3x3_flatten = delimited(
            opt(tag(b"\"".as_ref())),
            parse_2d_arr_3x3_flatten,
            opt(tag(b"\"".as_ref())),
        );
        let (_, latt) =
            alt((opt_quote_parse_2d_array, opt_quote_parse_2d_arr_3x3_flatten)).parse(latt)?;
        info.push(("Lattice".to_string(), latt));
    }
    info.push(("Properties".to_string(), prop_shape_value));
    Ok((input, (info, prop_shape)))
}

fn parse_natoms(input: &[u8]) -> IResult<&[u8], usize> {
    let (input, _) = complete::multispace0(input)?;
    let (input, natoms) = map_res(complete::digit1, |digits: &[u8]| {
        let s = std::str::from_utf8(digits).expect("digit1 expect ASCII");
        s.parse::<usize>()
    })
    .parse(input)?;
    let (input, _) = complete::multispace0(input)?;
    Ok((input, natoms))
}

fn parse_xyz_by_lines<'a>(
    input: &'a [u8],
    natoms_to_read: usize,
    prop_shape: &Vec<(&'a str, Ty, u8)>,
    arrs: &mut [(String, Value)],
) -> IResult<&'a [u8], usize> {
    let mut nat = 0;
    let mut proc_input = input;
    while !input.is_empty() && nat < natoms_to_read {
        let res = terminated(
            nom::bytes::streaming::take_until(&b"\n"[..]),
            streaming::newline,
        )
        .parse(proc_input);

        let (rest, line) = match res {
            Ok((rest, line)) => (rest, line),
            Err(nom::Err::Incomplete(_)) => {
                return Ok((input, nat));
            }
            Err(err) => return Err(err),
        };
        proc_input = rest;

        let (_, mut vs_raw) = delimited(
            multispace0,
            separated_list1(
                space1,
                alt((
                    recognize_float,
                    recognize_int,
                    recognize_bool,
                    // string is the least special element, so parsed in the end
                    recognize_bare_str,
                )),
            ),
            multispace0,
        )
        .parse(line)?;

        let mut loc = 0;
        for ((_, ty, n), (_, ref mut arr)) in prop_shape.iter().zip(arrs.iter_mut()) {
            match (ty, n, arr) {
                (_, 0, _) => unreachable!(),
                (Ty::I, 1, Value::VecInteger(v, _)) => {
                    let x = std::mem::take(&mut vs_raw[loc]);
                    let (_, x) = parse_int(x).expect("parse int");
                    let Value::Integer(x) = x else { unreachable!() };
                    v.push(x);
                    loc += 1;
                }
                (Ty::R, 1, Value::VecFloat(v, _)) => {
                    let x = std::mem::take(&mut vs_raw[loc]);
                    let (_, x) = parse_float(x).expect("parse float");
                    let Value::Float(x) = x else { unreachable!() };
                    v.push(x);
                    loc += 1;
                }
                (Ty::L, 1, Value::VecBool(v, _)) => {
                    let x = std::mem::take(&mut vs_raw[loc]);
                    let (_, x) = parse_bool(x).expect("parse bool");
                    let Value::Bool(x) = x else { unreachable!() };
                    v.push(x);
                    loc += 1;
                }
                (Ty::S, 1, Value::VecText(v, _)) => {
                    let x = std::mem::take(&mut vs_raw[loc]);
                    let (_, x) = parse_bare_str(x).expect("parse str");
                    let Value::Str(x) = x else { unreachable!() };
                    v.push(x);
                    loc += 1;
                }
                (Ty::I, nc, Value::MatrixInteger(m, _)) => {
                    let vv = vs_raw[loc..(loc + *nc as usize)]
                        .iter()
                        .map(|x| {
                            let (_, x) = parse_int(x).expect("parse float");
                            let Value::Integer(x) = x else { unreachable!() };
                            x
                        })
                        .collect::<Vec<_>>();
                    m.push(vv);
                    loc += *nc as usize;
                }
                (Ty::R, nc, Value::MatrixFloat(m, _)) => {
                    let vv = vs_raw[loc..(loc + *nc as usize)]
                        .iter()
                        .map(|x| {
                            let (_, x) = parse_float(x).expect("parse float");
                            let Value::Float(x) = x else { unreachable!() };
                            x
                        })
                        .collect::<Vec<_>>();
                    m.push(vv);
                    loc += *nc as usize;
                }
                (Ty::L, nc, Value::MatrixBool(m, _)) => {
                    let vv = vs_raw[loc..(loc + *nc as usize)]
                        .iter()
                        .map(|x| {
                            let (_, x) = parse_bool(x).expect("parse float");
                            let Value::Bool(x) = x else { unreachable!() };
                            x
                        })
                        .collect::<Vec<_>>();
                    m.push(vv);
                    loc += *nc as usize;
                }
                (Ty::S, nc, Value::MatrixText(m, _)) => {
                    let vv = vs_raw[loc..(loc + *nc as usize)]
                        .iter()
                        .map(|x| {
                            let (_, mut x) = parse_bare_str(x).expect("parse float");
                            let Value::Str(x) = std::mem::take(&mut x) else {
                                unreachable!()
                            };
                            x
                        })
                        .collect::<Vec<_>>();
                    m.push(vv);
                    loc += *nc as usize;
                }
                _ => unreachable!(),
            }
        }

        nat += 1;
    }

    Ok((proc_input, nat))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use extxyz_types::{Boolean, Integer};

    use crate::write_frame;

    use super::*;

    #[test]
    fn test_parse_properties() {
        let expect = b"species:S:1:pos:R:3";
        let (_, prop) = parse_properties(expect).unwrap();
        assert_eq!(prop[0], ("species", Ty::S, 1));
        assert_eq!(prop[1], ("pos", Ty::R, 3));

        // TODO: if size is 0, raise parsing error
    }

    #[test]
    fn test_promote_values_1d() {
        let mut vals = [];
        promote_values_1d(&mut vals).unwrap();

        assert!(vals.is_empty());

        let mut vals = [
            Value::Float(FloatNum::from(0.0)),
            Value::Float(FloatNum::from(0.0)),
            Value::Integer(Integer::from(1)),
        ];
        promote_values_1d(&mut vals).unwrap();

        assert!(matches!(vals[2], Value::Float(_)));

        let mut vals = [
            Value::Float(FloatNum::from(0.0)),
            Value::Float(FloatNum::from(0.0)),
            Value::Bool(Boolean::from(true)),
        ];
        assert!(promote_values_1d(&mut vals).is_err());
    }

    #[test]
    fn test_parse_1d_array() {
        let arr = b"[0,1]";
        let (_, val) = parse_1d_array(arr).unwrap();
        let Value::VecInteger(vs, 2) = val else {
            panic!("not a VecInteger")
        };
        assert_eq!(*vs[0], 0);
        assert_eq!(*vs[1], 1);

        let valid_expects: &[&[u8]] = &[
            b"[0.1, 0.2, 0]",
            // TODO: should support extra trailing ',' in the end of array
            // b"[0.1, 0.2, 0,]",
            b"[ 0.1, 0.2, 0.0]",
            b"[0.1, \t0.2, 0.0]",
            b"[0.1, 0.2,      0]",
            b"[0.1  , 0.2   , 0.0    ]",
        ];
        for expect in valid_expects {
            let (_, val) = parse_1d_array(expect).unwrap();
            let Value::VecFloat(vs, 3) = val else {
                panic!("not a VecFloat")
            };
            assert_eq!(*vs[0], 0.1);
            assert_eq!(*vs[1], 0.2);
            assert_eq!(*vs[2], 0.0);
        }
    }

    #[test]
    fn test_parse_2d_array() {
        let valid_expects: &[&[u8]] = &[
            b"[[-0,1],[2,2],[10,-1]]",
            b"[ [  -0, 1], \t[2,  2], [   10, -1]]",
            b"[[-0, 1  ], [ 2  , 2], [10 , -1]]",
            b"[[-0    \t , 1], [2, 2], [10, -1]]",
        ];
        for expect in valid_expects {
            let (_, val) = parse_2d_array(expect).unwrap();
            let Value::MatrixInteger(ms, (3, 2)) = val else {
                panic!("not a MatrixInteger")
            };
            assert_eq!(*ms[0][0], 0);
            assert_eq!(*ms[0][1], 1);
            assert_eq!(*ms[1][0], 2);
            assert_eq!(*ms[1][1], 2);
            assert_eq!(*ms[2][0], 10);
            assert_eq!(*ms[2][1], -1);
        }

        // TODO: test array of other types
    }

    #[test]
    fn test_parse_info_line_default() {
        let valid_expects: &[&[u8]] = &[
            b"key1=aa key2=bb",
            b"  key1=aa key2=bb",
            b"  key1=aa key2=bb  ",
            b"key1=aa  \t \t  key2=bb",
            b" key1 =aa key2=bb",
            b" key1= aa key2 =bb",
            b" key1  =  aa key2  =  bb",
            // TODO: move to final value test
            // b"key1= \"aa\" key2  =  \"bb\"",
        ];
        for expect in valid_expects {
            let (remain, v) = parse_info_line(expect).unwrap();
            assert!(remain.is_empty());
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[0].0).unwrap(),
                    str::from_utf8(v[0].1).unwrap()
                ),
                "key1=aa".to_string()
            );
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[1].0).unwrap(),
                    str::from_utf8(v[1].1).unwrap()
                ),
                "key2=bb".to_string()
            );
        }
    }

    #[test]
    fn test_parse_info_line_with_array() {
        let valid_expects: &[&[u8]] = &[
            b"key1=aa key2=bb Lattice=[[0,0,0],[10,4,4]]",
            b"key1=aa key2=bb Lattice=[[ 0,0 ,0],[10, 4,4]]",
            b"key1=aa key2=bb Lattice=[[0,0,0], [10,4,4]]",
        ];
        for expect in valid_expects {
            let (remain, v) = parse_info_line(expect).unwrap();
            assert!(remain.is_empty());
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[0].0).unwrap(),
                    str::from_utf8(v[0].1).unwrap()
                ),
                "key1=aa".to_string()
            );
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[1].0).unwrap(),
                    str::from_utf8(v[1].1).unwrap()
                ),
                "key2=bb".to_string()
            );
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[2].0).unwrap(),
                    str::from_utf8(v[2].1)
                        .unwrap()
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .collect::<String>()
                ),
                "Lattice=[[0,0,0],[10,4,4]]".to_string()
            );
        }
    }

    #[test]
    fn test_parse_info_line_with_str() {
        // key_value use recognize instead of doing further parse no extxyz::Value
        let valid_expects: &[&[u8]] = &[
            br#"key1=aa key2=bb pp=what"#,
            br#"key1=aa key2=bb pp= what"#,
        ];
        for expect in valid_expects {
            let (remain, v) = parse_info_line(expect).unwrap();
            assert!(remain.is_empty());
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[2].0).unwrap(),
                    str::from_utf8(v[2].1).unwrap(),
                ),
                "pp=what".to_string()
            );
        }

        let valid_expects: &[&[u8]] = &[
            br#"key1=aa key2=bb pp="what""#,
            br#"key1=aa key2=bb pp=  "what""#,
        ];
        for expect in valid_expects {
            let (remain, v) = parse_info_line(expect).unwrap();
            assert!(remain.is_empty());
            assert_eq!(
                format!(
                    "{}={}",
                    str::from_utf8(v[2].0).unwrap(),
                    str::from_utf8(v[2].1).unwrap(),
                ),
                "pp=\"what\"".to_string()
            );
        }
    }

    struct TFrame(Frame);

    impl std::fmt::Display for TFrame {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut buf = Vec::new();
            write_frame(&mut buf, &self.0).map_err(|_| std::fmt::Error)?;
            let s = std::str::from_utf8(&buf).map_err(|_| std::fmt::Error)?;
            f.write_str(s)
        }
    }

    #[test]
    fn test_parse_frame_default() {
        let inp = r#"2
Properties=species:S:1:pos:R:3 key1=aa key2=87 key3=thisisaverylongstring ZZPnonsense=65.9
Mn 0.0 0.5 0.5
C 0.0 0.5 0.3
"#;

        let mut rd = Cursor::new(inp.as_bytes());
        let frame = read_frame(&mut rd).unwrap();
        let frame = TFrame(frame);

        let expect = r#"2
ZZPnonsense=65.90000000 key1=aa key2=87 key3=thisisaverylongstring Properties=species:S:1:pos:R:3
Mn          0.00000000       0.50000000       0.50000000
C           0.00000000       0.50000000       0.30000000
"#;
        assert_eq!(format!("{frame}"), expect);
    }

    #[test]
    fn test_parse_frame_numeric_start_str_in_arrs() {
        let inp = r#"2
Properties=species:S:1:pos:R:3:s:S:1 key1=aa key2=87 key3=thisisaverylongstring ZZPnonsense=65.9
Mn 0.0 0.5 0.5 0000
C 0.0 0.5 0.3 878X
"#;

        let mut rd = Cursor::new(inp.as_bytes());
        let frame = read_frame(&mut rd).unwrap();
        let frame = TFrame(frame);

        let expect = r#"2
ZZPnonsense=65.90000000 key1=aa key2=87 key3=thisisaverylongstring Properties=species:S:1:pos:R:3:s:S:1
Mn          0.00000000       0.50000000       0.50000000 0000 
C           0.00000000       0.50000000       0.30000000 878X 
"#;
        assert_eq!(format!("{frame}"), expect);
    }

    #[test]
    fn test_parse_frame_without_properties() {
        let inp = r#"2
key1=aa key2=87 key3=thisisaverylongstring ZZPnonsense=65.9
Mn 0.0 0.5 0.5
C 0.0 0.5 0.3
"#;

        let mut rd = Cursor::new(inp.as_bytes());
        let frame = read_frame(&mut rd).unwrap();
        let frame = TFrame(frame);

        let expect = r#"2
ZZPnonsense=65.90000000 key1=aa key2=87 key3=thisisaverylongstring Properties=species:S:1:pos:R:3
Mn          0.00000000       0.50000000       0.50000000
C           0.00000000       0.50000000       0.30000000
"#;
        assert_eq!(format!("{frame}"), expect);
    }

    #[test]
    fn test_parse_lattice_from_flatten() {
        let inp = r#"3
Lattice="5.0 1.0 0.0 0.0 5.0 2.0 1.0 0.4 5.0" Properties=species:S:1:pos:R:3
Si    0.0    0.0    0.0
Si    2.5    2.5    2.5
O     1.25   1.25   1.25
"#;

        let mut rd = Cursor::new(inp.as_bytes());
        let frame = read_frame(&mut rd).unwrap();
        let frame = TFrame(frame);

        let expect = r#"3
Lattice=[[5.00000000, 0.00000000, 1.00000000], [1.00000000, 5.00000000, 0.40000000], [0.00000000, 2.00000000, 5.00000000]] Properties=species:S:1:pos:R:3
Si          0.00000000       0.00000000       0.00000000
Si          2.50000000       2.50000000       2.50000000
O           1.25000000       1.25000000       1.25000000
"#;
        assert_eq!(format!("{frame}"), expect);
    }

    #[test]
    fn test_no_equal_sign_line() {
        let inp = r#"3
full line that has no equal will be a comment line
Si    0.0    0.0    0.0
Si    2.5    2.5    2.5
O     1.25   1.25   1.25
"#;

        let mut rd = Cursor::new(inp.as_bytes());
        let frame = read_frame(&mut rd).unwrap();
        let frame = TFrame(frame);

        let expect = r#"3
comment="full line that has no equal will be a comment line" Properties=species:S:1:pos:R:3
Si          0.00000000       0.00000000       0.00000000
Si          2.50000000       2.50000000       2.50000000
O           1.25000000       1.25000000       1.25000000
"#;
        assert_eq!(format!("{frame}"), expect);
    }

    #[test]
    fn test_read_frames_default() {
        let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
        let mut rd = Cursor::new(inp.as_bytes());
        let mut frames = vec![];
        for frame in read_frames(&mut rd) {
            let frame = frame.unwrap();
            frames.push(frame);
        }

        assert_eq!(frames.len(), 2);
    }
}
