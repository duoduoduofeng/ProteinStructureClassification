import sys

def merge_files(proteins_file, cla_file, merged_data_file):

    fa_repres = {}
    with open(cla_file["filename"], 'r') as f1:
        row_num = 0
        for line in f1:
            row_num += 1
            if row_num < cla_file["start_row_num"]:
                continue
            
            parts = line.strip().split(' ')
            if len(parts) == cla_file["fields_count"]:
                fa_repres[parts[0]] = line.strip()


    with open(merged_data_file, 'w') as f_out:
        f_out.write(proteins_file['head'] + " " + cla_file['head'] + "\n")
        with open(proteins_file["filename"], 'r') as f2:
            row_num = 0
            for line in f2:
                row_num += 1
                if row_num < proteins_file["start_row_num"]:
                    continue
                
                parts = line.strip().split(' ')
                if len(parts) == proteins_file["fields_count"]:
                    if parts[0] in fa_repres:
                        f_out.write(line.strip() + " " + fa_repres[parts[0]] + '\n')
                    else:
                        # If there is no match, the protein only has superfamily, no family.
                        f_out.write(line.strip() + "\n")


    # with open(merged_data_file, 'w') as f_out:
    #     for line in merged_data.values():
    #         f_out.write(line + '\n')



meta_info = [
    {
        "filename": "../../data/scop-represented-structures-latest.txt",
        "start_row_num": 7,
        "fields_count": 3,
        "head": "DOMID REPRESENTED-PDBID REPRESENTED-PDBCHAIN",
        "example": "8000061 2DT5 B"
    },
    {
        "filename": "../../data/scop-cla-latest.txt",
        "start_row_num": 7,
        "fields_count": 11,
        "head": "FA-DOMID FA-PDBID FA-PDBREG FA-UNIID FA-UNIREG SF-DOMID SF-PDBID SF-PDBREG SF-UNIID SF-UNIREG SCOPCLA",
        "example": "8045703 3H8D C:1143-1264 Q64331 1143-1264 8091604 3H8D C:1143-1264 Q64331 1143-1264 TP=1,CL=1000003,CF=2001470,SF=3002524,FA=4004627"
    }
]

if __name__ == "__main__":
    merged_data_file_name = sys.argv[1]
    merge_files(meta_info[0], meta_info[1], merged_data_file_name)
