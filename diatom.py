import os
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"

    else:
        print
        "---  There is this folder!  ---"


M = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Pt']
dict = {'Sc': 2.11, 'Ti': 2.58, 'V': 2.72, 'Cr': 2.79, 'Mn': 3.06, 'Fe': 3.29, 'Co': 3.42, 'Ni': 3.40, 'Cu': 3.87,
        'Zn': 4.12, 'Pt': 0.00}
for i in range(0, 11):
    x1 = M[i]
    for j in range(i + 1, 11):
        x2 = M[j]

        X = [x1, x2]
        print(X)

        file = "O-dissociation/single-N/1-N/base/" + x1 + '-' + x2 + '/binding_energy'
        mkdir(file)  
        f = open('O-dissociation/single-N/1-N/base/' + x1 + '-' + x2 + '/INCAR', 'r')
        lines = f.readlines()  
        # for line in f_origin:
        f_new_incar = open('O-dissociation/single-N/1-N/base/' + x1 + '-' + x2 + '/binding_energy/INCAR', 'a')  
        for a in range(21):
            f_new_incar.write(lines[a])

        LDAUL = 'LDAUL     = 2     2'
        f_new_incar.write(LDAUL+'\n')
        LDAUU = 'LDAUU     = '
        for i in X:
            LDAUU = LDAUU + str(dict[i]) + '  '   
        f_new_incar.write(LDAUU+'\n')
        LDAUJ = 'LDAUJ     = 0.0   0.0'
        f_new_incar.write(LDAUJ+'\n')
        for a in range(25, 26):
            f_new_incar.write(lines[a])


        f = open('O-dissociation/single-N/1-N/base/' + x1 + '-' + x2 + '/CONTCAR', 'r')
        lines2 = f.readlines()  
        # for line in f_origin:
        f_new_poscar = open('O-dissociation/single-N/1-N/base/' + x1 + '-' + x2 + '/binding_energy/POSCAR', 'a')  
        for a in range(5):
            f_new_poscar.write(lines2[a])
        ele = ''
        for i in X:
            ele = ele + '   ' + i   
        ele = ele + '    '
        f_new_poscar.write(ele+'\n')
        num = '   1    1'
	f_new_poscar.write(num+'\n')
        for a in range(7, 10):
            f_new_poscar.write(lines2[a])
