/* native read using `nom`
*/
use extxyz_types::{DictHandler, FloatNum, Frame, Text, Value};
use nom::{
    self,
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::{
        self,
        complete::{multispace0, space0},
        streaming,
    },
    combinator::{all_consuming, map, map_res},
    multi::{many0, separated_list0, separated_list1},
    number,
    sequence::{delimited, separated_pair, terminated},
    IResult, Parser,
};
use std::{
    collections::HashMap,
    io::{self, BufRead},
};

fn _read_frame_native<R>(rd: &mut R, comment_override: Option<&str>) -> io::Result<Frame>
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

fn key_value(inp: &[u8]) -> IResult<&[u8], (&[u8], &[u8])> {
    let (inp, (k, v)) = separated_pair(
        delimited(
            multispace0,
            take_while1(|c: u8| c != b'=' && !c.is_ascii_whitespace()),
            multispace0,
        ),
        tag(&b"="[..]),
        delimited(
            multispace0,
            take_while1(|c: u8| !c.is_ascii_whitespace()),
            multispace0,
        ),
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
    number::complete::double
        .map(|i| Value::Float(i.into()))
        .parse(inp)
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
    let (linp, s) = take_while1(|c: u8| !c.is_ascii_whitespace() && c != b'"').parse(inp)?;
    let s = String::from_utf8(s.to_vec()).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(inp, nom::error::ErrorKind::Verify))
    })?;
    Ok((linp, Value::Str(Text::from(s))))
}

fn parse_quote_str(inp: &[u8]) -> IResult<&[u8], Value> {
    let parse_inner = map(
        many0(alt((
            take_while1(|b| b != b'\\' && b != b'"'), // raw bytes
            map(tag(r#"\""#), |_| b"\"".as_ref()),
            map(tag(r#"\\"#), |_| b"\\".as_ref()),
            map(tag(r#"\n"#), |_| b"\n".as_ref()),
        ))),
        |chunks: Vec<&[u8]>| {
            let s = chunks.concat();
            let s = String::from_utf8(s).unwrap();
            Value::Str(Text::from(s))
        },
    );

    delimited(tag(b"\"".as_ref()), parse_inner, tag(b"\"".as_ref())).parse(inp)
}

fn parse_value(inp: &[u8]) -> IResult<&[u8], Value> {
    // order conform with the spec, see README for spec definition.
    let (inp, v) = delimited(
        multispace0,
        alt((
            // float before int, to avoid 3.14 -> 3
            parse_float,
            parse_int,
            // bool comes before str, to avoid boll true -> str "true"
            parse_bool,
            parse_bare_str,
            parse_quote_str,
        )),
        multispace0,
    )
    .parse(inp)?;
    Ok((inp, v))
}

fn parse_1d_array(inp: &[u8]) -> IResult<&[u8], Value> {
    let (inp_, mut vals) = delimited(
        tag(b"[".as_ref()),
        separated_list0(tag(b",".as_ref()), parse_value),
        tag(b"]".as_ref()),
    )
    .parse(inp)?;

    // promote
    // TODO: this need to be a verbose error
    if promote_values_1d(&mut vals).is_err() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            inp,
            nom::error::ErrorKind::Verify,
        )));
    }

    if vals.is_empty() {
        // empty Vec, doesnt matter use which Value::Vec**
        Ok((inp_, Value::VecBool(Vec::new(), 0)))
    } else {
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
}

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

fn parse_info_line(inp: &[u8]) -> IResult<&[u8], HashMap<String, Value>> {
    let (inp, _) = space0.parse(inp)?;
    let (inp, vec_kv) = separated_list1(space0, key_value).parse(inp)?;
    let (inp, _) = space0.parse(inp)?;
    let mut kv = HashMap::new();
    for (k, v) in vec_kv {
        let key = String::from_utf8(k.to_vec()).expect("invalid utf8");
        let (_, val) = parse_value(v)?;

        let old_val = kv.insert(key, val);
        if old_val.is_some() {
            return Err(nom::Err::Failure(nom::error::Error::new(
                k,
                nom::error::ErrorKind::Verify,
            )));
        }
    }
    Ok((inp, kv))
}

fn parse_frame(input: &[u8]) -> IResult<&[u8], Frame> {
    let (input, _) = streaming::space0(input)?;
    let (input, natoms) = map_res(streaming::digit1, |digits: &[u8]| {
        let s = std::str::from_utf8(digits).expect("digit1 expect ASCII");
        s.parse::<u32>()
    })
    .parse(input)?;
    let (mut input, line) = terminated(
        nom::bytes::streaming::take_until(&b"\n"[..]),
        streaming::newline,
    )
    .parse(input)?;
    let (_, info) = all_consuming(parse_info_line).parse(line)?;

    // TODO: check "properties"/"property"/"Property" and raise error with help message.
    // TODO: check "lattice" and raise error with help message.

    let maybe_prop = info.get("Properties");
    let maybe_latt = info.get("Lattice");

    let mut atom_lines = Vec::new();
    for i in 0..natoms {
        let (rest, line) = terminated(
            nom::bytes::streaming::take_until(&b"\n"[..]),
            streaming::newline,
        )
        .parse(input)?;
        atom_lines.push(line);
        // bring the rest out as remaining input
        input = rest;
    }

    Ok((
        input,
        Frame {
            natoms,
            info: DictHandler(Vec::new()),
            arrs: DictHandler(Vec::new()),
        },
    ))
}

#[cfg(test)]
mod tests {
    use extxyz_types::{Boolean, Integer};

    use super::*;

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
            b"key1= \"aa\" key2  =  \"bb\"",
        ];
        for expect in valid_expects {
            let (remain, v) = parse_info_line(expect).unwrap();
            assert!(remain.is_empty());
            assert_eq!(format!("{}", v.get("key1").unwrap()), "aa".to_string());
            assert_eq!(format!("{}", v.get("key2").unwrap()), "bb".to_string());
        }
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
        // TODO:
        // let arr = b"[]";
        // let (_, _) = parse_1d_array(arr).unwrap();

        let valid_expects: &[&[u8]] = &[
            b"[0.1, 0.2, 0]",
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
}
