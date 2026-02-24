/* native read using `nom`
*/
use extxyz_types::{DictHandler, Frame, Value};
use nom::{
    self, IResult, Parser, branch::alt, bytes::streaming::{take_until, take_while1}, character::streaming, combinator::{map_res, opt}, multi::many0, sequence::{separated_pair, terminated}
};
use std::{collections::HashMap, io::{self, BufRead}};

// fn wspace_to_eol(input: &str) -> IResult<&str, &str> {
//     let (inp, _) = space0(input)?;
//     comment_or_eol(inp)
// }
// fn wspace_any(input: &str) -> IResult<&str, &str> {
//     let (inp, _) = many0(wspace_to_eol).parse(input)?;
//     space0(inp)
// }
// fn wspace_lines(input: &str) -> IResult<&str, &str> {
//     let (inp, _) = opt(comment).parse(input)?;
//     let (inp, _) = space0(inp)?;
//     let (inp, _) = line_ending(inp)?;
//     let (inp, _) = many0(wspace_to_eol).parse(inp)?;
//     Ok((inp, ""))
// }
// fn wspace(input: &str) -> IResult<&str, &str> {
//     let (inp, _) = alt((space1, line_ending)).parse(input)?;
//     wspace_any(inp)
// }

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

#[derive(Debug)]
struct Info {
    kv: HashMap<String, Value>,
}

fn parse_info_line(line: &[u8]) -> IResult<&[u8], Info> {
    // separated_pair(take_while1(cond) sep, second)
    todo!()
}

fn parse_frame(input: &[u8]) -> IResult<&[u8], Frame> {
    let (input, _) = streaming::space0(input)?;
    let (input, natoms) = map_res(streaming::digit1, |digits: &[u8]| {
        let s = std::str::from_utf8(digits).expect("digit1 expect ASCII");
        s.parse::<u32>()
    })
    .parse(input)?;
    let (mut input, line) = terminated(take_until(&b"\n"[..]), streaming::newline).parse(input)?;

    let mut atom_lines = Vec::new();
    for i in 0..natoms {
        let (rest, line) = terminated(take_until(&b"\n"[..]), streaming::newline).parse(input)?;
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
