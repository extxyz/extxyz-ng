/* native read parsed using `nom`
*/
use extxyz_types::{DictHandler, FloatNum, Frame, Text, Value};
use nom::{
    self,
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::{
        self,
        complete::{multispace0, space0, space1},
        streaming,
    },
    combinator::{all_consuming, map, map_res, recognize, verify},
    multi::{many0, separated_list0, separated_list1},
    number,
    sequence::{delimited, separated_pair, terminated},
    IResult, Parser,
};
use std::{
    collections::{BTreeMap, HashMap},
    io::{self, BufRead},
};

pub(crate) fn _read_frame_native<R>(rd: &mut R, comment_override: Option<&str>) -> io::Result<Frame>
where
    R: BufRead,
{
    loop {
        let buf = rd.fill_buf()?;
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF reached before parsing frame",
            ));
        }

        match parse_frame(buf) {
            Ok((remaining, mut frame)) => {
                let amount = buf.len() - remaining.len();
                rd.consume(amount);
                if let Some(comment) = comment_override {
                    frame.set_comment(comment);
                }

                return Ok(frame);
            }
            Err(nom::Err::Incomplete(_needed)) => {
                let len = buf.len();
                rd.consume(len);

                continue;
            }
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        }
    }
}

// XXX: can I lru cache this call?
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

// i32
fn parse_int(inp: &[u8]) -> IResult<&[u8], Value> {
    character::complete::i32
        .map(|i| Value::Integer(i.into()))
        .parse(inp)
}

fn parse_float(inp: &[u8]) -> IResult<&[u8], Value> {
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
    let (inp, float) = number::complete::double
        .map(|i| Value::Float(i.into()))
        .parse(inp)?;

    Ok((inp, float))
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

fn parse_bare_str(inp: &[u8]) -> IResult<&[u8], Value> {
    let (linp, s) = take_while1(|c: u8| c.is_ascii_alphanumeric() || c == b'_').parse(inp)?;
    if !s[0].is_ascii_alphabetic() && s[0] != b'_' {
        return Err(nom::Err::Error(nom::error::Error::new(
            linp,
            nom::error::ErrorKind::Verify,
        )));
    }
    let s = String::from_utf8(s.to_vec()).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
    })?;
    Ok((linp, Value::Str(Text::from(s))))
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
    Ok((remain, Value::Str(Text::from(s))))
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
        parse_bare_properties_str,
        parse_bare_str,
        parse_quote_str,
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

fn parse_frame(input: &[u8]) -> IResult<&[u8], Frame> {
    let (input, _) = streaming::multispace0(input)?;
    let (input, natoms) = map_res(streaming::digit1, |digits: &[u8]| {
        let s = std::str::from_utf8(digits).expect("digit1 expect ASCII");
        s.parse::<usize>()
    })
    .parse(input)?;
    let (input, _) = streaming::multispace0(input)?;
    // let (input, _) = streaming::newline(input)?;

    let (mut input, line) = terminated(
        nom::bytes::streaming::take_until(&b"\n"[..]),
        streaming::newline,
    )
    .parse(input)?;

    let (_, info_kv) = all_consuming(parse_info_line).parse(line)?;

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

    // XXX: latt and prop_shape better to be stored separatly from pure_kv
    let mut info = Vec::with_capacity(kv.len() + 2);
    for (k, v) in kv {
        let (_, v) = parse_kv_right(v)?;
        info.push((String::from_utf8(k.to_vec()).expect("utf8"), v));
    }

    // TODO: latt can be a matrix wrapped inside a pair of double quotes, test it
    if let Some(latt) = maybe_latt {
        let (_, latt) = parse_2d_array(latt)?;
        info.push(("Lattice".to_string(), latt));
    }
    info.push(("Properties".to_string(), prop_shape_value));

    // TODO: validate natoms and number of rows of the arr

    let mut mhs: HashMap<&str, Vec<Value>> = HashMap::with_capacity(prop_shape.len());
    for _i in 0..natoms {
        let (rest, line) = terminated(
            nom::bytes::streaming::take_until(&b"\n"[..]),
            streaming::newline,
        )
        .parse(input)?;
        let (_, mut vs_raw) = delimited(
            multispace0,
            separated_list1(
                space1,
                alt((parse_bare_str, parse_float, parse_int, parse_bool)),
            ),
            multispace0,
        )
        .parse(line)?;

        let mut loc: u8 = 0;
        for (name, ty, n) in &prop_shape {
            if *n == 0 {
                // TODO: verbose context error
                return Err(nom::Err::Failure(nom::error::Error::new(
                    rest,
                    nom::error::ErrorKind::Verify,
                )));
            }
            if *n == 1 {
                let vv = std::mem::take(&mut vs_raw[loc as usize]);
                mhs.entry(name).or_default().push(vv);
                loc += *n;
            } else {
                let mut vv: Vec<Value> = Vec::with_capacity(*n as usize);
                vv.resize_with(*n as usize, Default::default);
                for i in 0..(*n) {
                    vv[i as usize] = std::mem::take(&mut vs_raw[(i + loc) as usize]);
                }
                let vxs = match ty {
                    Ty::I => {
                        let vv = vv
                            .into_iter()
                            .map(|vv| {
                                let Value::Integer(inner) = vv else {
                                    // actual value not consistent with ty in prop_shape
                                    return Err(nom::Err::Failure(nom::error::Error::new(
                                        rest,
                                        nom::error::ErrorKind::Verify,
                                    )));
                                };
                                Ok(inner)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let len = vv.len();
                        Value::VecInteger(vv, len as u32)
                    }
                    Ty::R => {
                        let vv = vv
                            .into_iter()
                            .map(|vv| {
                                let inner = match vv {
                                    Value::Integer(int_inner) => {
                                        FloatNum::from(f64::from(*int_inner))
                                    }
                                    Value::Float(inner) => inner,
                                    _ => {
                                        // actual value not consistent with ty in prop_shape
                                        return Err(nom::Err::Failure(nom::error::Error::new(
                                            rest,
                                            nom::error::ErrorKind::Verify,
                                        )));
                                    }
                                };
                                Ok(inner)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let len = vv.len();
                        Value::VecFloat(vv, len as u32)
                    }
                    Ty::L => {
                        let vv = vv
                            .into_iter()
                            .map(|vv| {
                                let Value::Bool(inner) = vv else {
                                    // actual value not consistent with ty in prop_shape
                                    return Err(nom::Err::Failure(nom::error::Error::new(
                                        rest,
                                        nom::error::ErrorKind::Verify,
                                    )));
                                };
                                Ok(inner)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let len = vv.len();
                        Value::VecBool(vv, len as u32)
                    }
                    Ty::S => {
                        let vv = vv
                            .into_iter()
                            .map(|vv| {
                                let Value::Str(inner) = vv else {
                                    // actual value not consistent with ty in prop_shape
                                    return Err(nom::Err::Failure(nom::error::Error::new(
                                        rest,
                                        nom::error::ErrorKind::Verify,
                                    )));
                                };
                                Ok(inner)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let len = vv.len();
                        Value::VecText(vv, len as u32)
                    }
                };
                mhs.entry(name).or_default().push(vxs);
                loc += *n;
            }
        }

        // bring the rest out as remaining input
        input = rest;
    }
    debug_assert!(input.is_empty());

    // TODO: zero-copy
    let mut arrs: Vec<(String, Value)> = Vec::with_capacity(prop_shape.len());
    for (name, ty, n) in &prop_shape {
        match (ty, n) {
            (_, 0) => unreachable!(),
            (Ty::I, 1) => {
                let vinner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let vinner = vinner
                    .iter()
                    .map(|i| {
                        let Value::Integer(x) = i else { unreachable!() };
                        *x
                    })
                    .collect::<Vec<_>>();
                let vinner = Value::VecInteger(vinner, natoms as u32);
                arrs.push((name.to_string(), vinner));
            }
            (Ty::R, 1) => {
                let vinner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let vinner = vinner
                    .iter()
                    .map(|i| {
                        let Value::Float(x) = i else { unreachable!() };
                        *x
                    })
                    .collect::<Vec<_>>();
                let vinner = Value::VecFloat(vinner, natoms as u32);
                arrs.push((name.to_string(), vinner));
            }
            (Ty::L, 1) => {
                let vinner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let vinner = vinner
                    .iter()
                    .map(|i| {
                        let Value::Bool(x) = i else { unreachable!() };
                        *x
                    })
                    .collect::<Vec<_>>();
                let vinner = Value::VecBool(vinner, natoms as u32);
                arrs.push((name.to_string(), vinner));
            }
            (Ty::S, 1) => {
                let vinner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let vinner = vinner
                    .iter()
                    .map(|i| {
                        let Value::Str(x) = i else { unreachable!() };
                        // TODO: ?? zero-copy?
                        x.clone()
                    })
                    .collect::<Vec<_>>();
                let vinner = Value::VecText(vinner, natoms as u32);
                arrs.push((name.to_string(), vinner));
            }
            (Ty::I, nc) => {
                let minner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let minner = minner
                    .iter()
                    .map(|i| {
                        let Value::VecInteger(vi, nx) = i else {
                            unreachable!()
                        };
                        debug_assert_eq!(*nx, *nc as u32);
                        vi.clone()
                    })
                    .collect::<Vec<_>>();
                let minner = Value::MatrixInteger(minner, (natoms as u32, *nc as u32));
                arrs.push((name.to_string(), minner));
            }
            (Ty::R, nc) => {
                let minner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let minner = minner
                    .iter()
                    .map(|i| {
                        let Value::VecFloat(vi, nx) = i else {
                            unreachable!()
                        };
                        debug_assert_eq!(*nx, *nc as u32);
                        vi.clone()
                    })
                    .collect::<Vec<_>>();
                let minner = Value::MatrixFloat(minner, (natoms as u32, *nc as u32));
                arrs.push((name.to_string(), minner));
            }
            (Ty::L, nc) => {
                let minner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let minner = minner
                    .iter()
                    .map(|i| {
                        let Value::VecBool(vi, nx) = i else {
                            unreachable!()
                        };
                        debug_assert_eq!(*nx, *nc as u32);
                        vi.clone()
                    })
                    .collect::<Vec<_>>();
                let minner = Value::MatrixBool(minner, (natoms as u32, *nc as u32));
                arrs.push((name.to_string(), minner));
            }
            (Ty::S, nc) => {
                let minner = mhs
                    .get(name)
                    .expect("key not found from where it must exist");
                let minner = minner
                    .iter()
                    .map(|i| {
                        let Value::VecText(vi, nx) = i else {
                            unreachable!()
                        };
                        debug_assert_eq!(*nx, *nc as u32);
                        vi.clone()
                    })
                    .collect::<Vec<_>>();
                let minner = Value::MatrixText(minner, (natoms as u32, *nc as u32));
                arrs.push((name.to_string(), minner));
            }
        }
    }

    let frame = Frame {
        natoms: natoms as u32,
        info: DictHandler(info),
        arrs: DictHandler(arrs),
    };

    Ok((input, frame))
}

#[cfg(test)]
mod tests {
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
        let inp = &br#"2
Properties=species:S:1:pos:R:3 key1=aa key2=87 key3=thisisaverylongstring ZZPnonsense=65.9
Mn 0.0 0.5 0.5
C 0.0 0.5 0.3
"#[..];

        let (_, frame) = parse_frame(inp)
            .map_err(|err| err.map_input(|inp| str::from_utf8(inp).unwrap()))
            .unwrap();
        let frame = TFrame(frame);

        let expect = r#"2
ZZPnonsense=65.90000000 key1=aa key2=87 key3=thisisaverylongstring Properties=species:S:1:pos:R:3
Mn          0.00000000       0.50000000       0.50000000
C           0.00000000       0.50000000       0.30000000
"#;
        assert_eq!(format!("{frame}"), expect);
    }

    #[test]
    fn test_parse_frame_without_properties() {
        let inp = &br#"2
key1=aa key2=87 key3=thisisaverylongstring ZZPnonsense=65.9
Mn 0.0 0.5 0.5
C 0.0 0.5 0.3
"#[..];

        let (_, frame) = parse_frame(inp)
            .map_err(|err| err.map_input(|inp| str::from_utf8(inp).unwrap()))
            .unwrap();
        let frame = TFrame(frame);

        let expect = r#"2
ZZPnonsense=65.90000000 key1=aa key2=87 key3=thisisaverylongstring Properties=species:S:1:pos:R:3
Mn          0.00000000       0.50000000       0.50000000
C           0.00000000       0.50000000       0.30000000
"#;
        assert_eq!(format!("{frame}"), expect);
    }
}
