
import glob
import shutil,os

#创建文件路径
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        b = 1

phase = 'ART'
img_path = r'D:\BaiduNetdiskDownload\Eunice\Internal Validation\*\*/'#存放png图片的地址，有很多张二维图片
NC_nrrd_address = glob.glob(img_path + '*'+ phase +'.nrrd')
nrrd_address = glob.glob(img_path + '*'+ phase +'.nrrd')
#liver_nrrd_address = glob.glob(img_path + '*NC_liver.nrrd')
nii_address = glob.glob(img_path + '*PV.nii')
new_path = 'F:\\MVIdata\\intertest\\'

m = 0
for i in range(len(nrrd_address)):
    nclist = NC_nrrd_address[i].split('\\')
    list1 = nrrd_address[i].split('_')
    list2 = list1[-1].split('.')
    list0 = nii_address[i].split('\\')
    #list4 = liver_nrrd_address[i].split('\\')
    list3 = nrrd_address[i].split('\\')
    #if (list0[4] == list3[4]) & (list0[4] ==list4[4])  :
    if (list0[4] == list3[4]) :
        new_nrrd_dir = new_path + list3[4]+'\\'+list3[5]+'\\'+list3[4]+'_'+list3[5]+'_'+list2[0]+'_tumormask'+'.nrrd'
        #new_liver_nrrd_dir = new_path  + list3[4] + '\\' + list3[5] + '\\' + list3[4] + '_' + list3[5] + '_' + list2[0] + '_livermask' + '.nrrd'
        new_nii_dir = new_path + list3[4]+'\\'+list3[5]+'\\'+list3[4]+'_'+list3[5]+'_'+list2[0]+'_raw'+'.nii'
        New_file_path = new_path + list3[4]+'\\'+list3[5]
        mkdir(New_file_path)
        shutil.copy(nrrd_address[i], new_nrrd_dir)
        shutil.copy(nii_address[i], new_nii_dir)
        #shutil.copy(liver_nrrd_address[i], new_liver_nrrd_dir)
    else:
        print('nc:',)
        print('nii:',nii_address[i])
        #print('livernrrd:',liver_nrrd_address[i])
        print('tumornrrd:', nrrd_address[i])
    m+=1
    print(m)
