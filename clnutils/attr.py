# lithological order
Lith_order = ["K1","K2","K3","K4","K5", # intrusions
              "P1","P1B","P2","P3","P5","P7","P8", # intrusions
              "PHBX","QABX","BXTO","BX", # breccias
              "H1","H1B","CGL","SS","SARG","SSC", # wall rock
              "SSL","SSM","SSN","VABX","VALT","VATF", # wall rock
              "UNCL","UNID","IU", # unidentified
              "FLTZ","FLTH","MYLO","BSZ","MTF","SNLS","NREC", # additional codes
              "FG", # unique to Sulpherets, see email from Ross Hammett
              "STF","STF2","WR", # in Alice's list
              "No Data"]

# replacement names
lith_name_map = {"SKA":"BSZ",
                 "CVN":"FLTZ",
                 "QSBX":"QABX",
                 "P4":"P2",
                 "QVN":"FLTH",
                 "SVN":"MYLO",
                 "DDRT":"K4",
                 "PMFP":"K3",
                 "PPFP":"P3",# ask, multiple possible
                 "PMON":"P8",
                 "VU":"VABX",# ask, multiple possible
                 "PSYN":"P8",
                 "QTVN":"P1",
                 "VCGL":"CGL"}