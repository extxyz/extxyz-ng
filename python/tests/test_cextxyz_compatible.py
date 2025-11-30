from extxyz import cextxyz

def test_default(tmp_path):
    inp_xyz = """4
key1=a key2=a/b key3=a@b key4="a@b" 
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"""
    with open(tmp_path / "inp.xyz", mode="w+") as f:
        _ = f.write(inp_xyz)
        f.flush()

        fp = cextxyz.cfopen(str(tmp_path / "inp.xyz"), "r")
        nat, info, arrs = cextxyz.read_frame_dicts(fp)

        cextxyz.cfclose(fp)

        fp = cextxyz.cfopen(str(tmp_path / "out.xyz"), "w")
        cextxyz.write_frame_dicts(fp, nat, info, arrs)

        cextxyz.cfclose(fp)

    with open(tmp_path / "out.xyz", mode="r") as f:
        s = f.read()
        out = """4
key1=a key2=a/b key3=a@b key4=a@b Properties=species:S:1:pos:R:3
Mg        -4.25650000       3.79180000      -2.54123000
C        -1.15405000       2.86652000      -1.26699000
C        -5.53758000       3.70936000       0.63504000
C        -7.28250000       4.71303000      -3.82016000
"""
        assert s == out
