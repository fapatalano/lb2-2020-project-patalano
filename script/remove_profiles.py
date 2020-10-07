#!/bin/python3

def remove_profiles(profile):
    tot=0.0
    with open(profile) as file:
        for line in file:
            line=line.strip().split()
            if len(line)!=0:
                line=list(map(float,line[2:]))
                tot=sum(line,tot)

    return tot


if __name__ == '__main__':
    # profile= sys.argv[1]
    # remove_profiles(profile)
    import os
    import sys
    try :
        dir_path= sys.argv[1]
        for filename in sorted(os.listdir(dir_path)):
            profile= dir_path+'/'+filename
            tot=remove_profiles(profile)
            if tot==0.0: print(profile)
    except IndexError:
        print("\nMissing argument: directory path")
