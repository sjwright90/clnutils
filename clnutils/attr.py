# lithological order
Lith_order = [
    "K1",
    "K2",
    "K3",
    "K4",
    "K5",  # intrusions
    "P1",
    "P1B",
    "P2",
    "P3",
    "P5",
    "P7",
    "P8",  # intrusions
    "PHBX",
    "QABX",
    "BXTO",
    "BX",  # breccias
    "H1",
    "H1B",
    "CGL",
    "SS",
    "SARG",
    "SSC",  # wall rock
    "SSL",
    "SSM",
    "SSN",
    "VABX",
    "VALT",
    "VATF",  # wall rock
    "UNCL",
    "UNID",
    "IU",  # unidentified
    "FLTZ",
    "FLTH",
    "MYLO",
    "BSZ",
    "MTF",
    "SNLS",
    "NREC",  # additional codes
    "FG",  # unique to Sulpherets, see email from Ross Hammett
    "STF",
    "STF2",
    "WR",  # in Alice's list
    "No Data",
]

# replacement names
lith_name_map = {
    "SKA": "BSZ",
    "CVN": "FLTZ",
    "QSBX": "QABX",
    "P4": "P2",
    "QVN": "FLTH",
    "SVN": "MYLO",
    "DDRT": "K4",
    "PMFP": "K3",
    "PPFP": "P3",  # ask, multiple possible
    "PMON": "P8",
    "VU": "VABX",  # ask, multiple possible
    "PSYN": "P8",
    "QTVN": "P1",
    "VCGL": "CGL",
}

# replacement characters
super_sub_scriptreplace = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
}
