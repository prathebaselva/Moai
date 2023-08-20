import numpy as np
import os
import sys

def convertnpyToPLY(npyfile, filename, outputfolder):
    points = np.load(npyfile)
    convertToPLY(points, filename, outputfolder)


def normalizeMesh(nparray):
    maxdim = np.max(nparray, axis=0)
    mindim = np.min(nparray, axis=0)
    m = maxdim - mindim
    center = (maxdim+mindim)/2
    radius = np.max(m)
    norm_points = (nparray - center)/radius
    return norm_points

def convertToPLY(points, filename, outputfolder):
    plyfile = open(os.path.join(outputfolder,filename), "w")
    points = normalizeMesh(points)

    plyfile.write("ply\n")
    plyfile.write("format ascii 1.0\n")
    plyfile.write("element vertex "+str(len(points))+"\n")
    plyfile.write("property float x\n")
    plyfile.write("property float y\n")
    plyfile.write("property float z\n")
    plyfile.write("element face 0\n")
    plyfile.write("property list uchar int vertex_indices\n")
    plyfile.write("end_header\n")
    for p in points:
        plyfile.write(str(p[0])+" "+str(p[1])+" "+str(p[2])+"\n")
    plyfile.close()


if __name__ == '__main__':
    convertToPLY(*sys.argv[1:])

