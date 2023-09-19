import numpy as np
import os
import sys
from glob import glob




def facescape(imgfolder, dpmapfolder, outputfolder, datasetname):
    npy = {}
    npyfiles = {}
    for imgpath in sorted(glob(imgfolder+'/*')):
        img = imgpath.split('/')[-1]
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):

            #if not ("1_neutral" in img):
            #    continue
            #print(img)
            actorpath = (imgpath[len(imgfolder):])
            actordata = (actorpath.split('.')[0]).split('_')
            #print(actordata)
            actor = '_'.join([actordata[i] for i in range(len(actordata)-1)])
            #print(actor)
            #print(actorpath)
            #exit()
            #actor = (imgpath[len(imgfolder):]).split('/')[0]
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)
    
    for actor in npy.keys():
        dpmappath = os.path.join(dpmapfolder, actor+'.png')
        if os.path.exists(dpmappath):
            npyfiles[actor].append(actor+'.png')
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)


def coma(imgfolder, paramfolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    count = 0
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        data = img.split('.')
        expression = data[0]
        index = data[1]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            print(actorpath)
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            actor = actor+'__'+expression+'__'+index
            #print(actor)
            #exit()

            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)

    
    for actor in npy.keys():
        data = actor.split('__')
        print(data)
        actorname = data[0]
        expression = data[1]
        index = data[2]
        #plypath = os.path.join(paramfolder, actorname, expression, expression+'.'+index+'.ply')
        npypath = os.path.join(paramfolder, actorname, expression, expression+'.'+index+'.npy')
        #if os.path.exists(plypath):
        if os.path.exists(npypath):
            npyfiles[actor].append(npypath)
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]
        #print(npyfiles[actor])
        #exit()

    print(len(npyfiles))
    #print(npyfiles)
    #exit()
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def tempeh(imgfolder, paramfolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    count = 0
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        data = img.split('.')
        expression = data[0]
        index = data[1]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            print(actorpath)
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            actor = actor+'__'+expression+'__'+index

            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)
    
    for actor in npy.keys():
        data = actor.split('__')
        #print(data)
        actorname = data[0]
        expression = data[1]
        index = data[2]
        #plypath = os.path.join(paramfolder, actorname, expression, expression+'.'+index+'.ply')
        npzpath = os.path.join(paramfolder, actorname, expression, expression+'.'+index+'.npz')
        if os.path.exists(npzpath):
            npyfiles[actor].append(npzpath)
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def moai(imgfolder, metafolder, lmkfolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    count = 0
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        imgindex = img.split('.')[0]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            print(actorpath)
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            print(actor)
            metapath = os.path.join(metafolder, 'metadata_'+actor+'_'+str(imgindex)+'.json')
            lmkpath = os.path.join(lmkfolder, 'ldmks_'+actor+'_'+str(imgindex)+'.json')
            print(metapath)
            print(lmkpath)
            #exit()
            if os.path.exists(imgpath) and os.path.exists(metapath) and os.path.exists(lmkpath):
                print("yes")
            else:
                print("no")

            if os.path.exists(imgpath) and os.path.exists(metapath) and os.path.exists(lmkpath):
                if actor not in npy:
                    npy[actor] = []
                    npyfiles[actor] = []

                npy[actor].append((imgpath, metapath, lmkpath))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npy)

def get_image_paths(imgfolder, metafolder, lmkfolder, outputfolder, datasetname):
    if datasetname == 'MOAI':
        moai(imgfolder, metafolder, lmkfolder, outputfolder, datasetname)
        
if __name__ == '__main__':
    get_image_paths(*sys.argv[1:])
