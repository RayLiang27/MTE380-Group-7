print("Hello World!")
import numpy as np
from scipy.special import sph_harm
from scipy.optimize import fsolve
# importing libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# Example: Plotting a triangle and a circle in 3D
#import numpy as np
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


#test_sph3(6)
l=10
l_1=10
l_11=8
l_12=8
l_2=10
l_21=8
l_22=8
l_3=10
l_31=8
l_32=8

# Define two vectors
phi = 70 * np.pi / 180
alpha = 50 * np.pi / 180
nrm = np.array([np.cos(phi)*np.cos(alpha), np.cos(phi)*np.sin(alpha), np.sin(phi)])
S = np.array([0, 0, 17.05])

def position_and_orientation(nrm, S):
    vector_i = np.array([1, 0, 0])
    # Calculate the cross product
    cross_product = np.cross(nrm, vector_i)
    anorm = np.linalg.norm(cross_product)
    # Normalize the vector
    v_hat = cross_product / anorm
    print("v_hat",v_hat)
    a = l / (2*np.cos(30 * np.pi / 180))
    cross_product = np.cross(v_hat, nrm)
    # Calculate the norm (magnitude) of the vector
    anorm = np.linalg.norm(cross_product)
    # Normalize the vector
    u_hat = cross_product / anorm
    w_hat = np.cross(u_hat, v_hat)
    print("u_hat",u_hat,"v_hat",v_hat,"w_hat",w_hat)
    print("croos prod u_hat and v_hat",np.dot(u_hat,v_hat))
    print("croos prod v_hat and w_hat",np.dot(v_hat,w_hat))
    print("croos prod w_hat and u_hat",np.dot(w_hat,u_hat))
    # Calculate the cross product
    x1 = S + a * v_hat
    x2 = S - a * np.sin(30. * np.pi / 180.0) * v_hat + a * np.cos(30. * np.pi / 180.) * u_hat
    x3 = S - a * np.sin(30. * np.pi / 180.0) * v_hat - a * np.cos(30. * np.pi / 180.) * u_hat
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
    }
# Example usage
result = position_and_orientation(nrm, S)
x1 = result["x1"]
x2 = result["x2"]
x3 = result["x3"]         
print("x1:", x1)
print("x2:", x2)
print("x3:", x3)
print("mag",np.sqrt((x1[0]-S[0])**2+(x1[1]-S[1])**2+(x1[2]-S[2])**2),np.sqrt((x2[0]-S[0])**2+(x2[1]-S[1])**2+(x2[2]-S[2])**2),np.sqrt((x3[0]-S[0])**2+(x3[1]-S[1])**2+(x3[2]-S[2])**2))
print("mag2",np.sqrt((x1[0]-S[0])**2+(x1[1]-S[1])**2),np.sqrt((x2[0]-S[0])**2+(x2[1]-S[1])**2),np.sqrt((x3[0]-S[0])**2+(x3[1]-S[1])**2))
print("ratio1",x1[0]/x1[1],np.arctan(1/(x2[0]/x2[1])),np.arctan(1/(x3[0]/x3[1])))
print("croos prod x1 and nrm",np.dot(S-x1,nrm))
print("croos prod x2 and nrm",np.dot(S-x2,nrm))
print("croos prod x3 and nrm",np.dot(S-x3,nrm))


def calculate_vectors_and_angles_1(l, l_1, l_11, l_12, x1,x2,x3):

    # Calculating theta_11
    # Calculate numerator
    AA_tmp =  l_12**2 - (l_1 - x1[1])**2 - l_11**2 - x1[2]**2
    #Calculate denominator
    bb_tmp = (2 * (l_1 - x1[1]) * l_11)
    cc_tmp = (2 * x1[2] * l_11)
    denom_AA = np.sqrt(bb_tmp**2 + cc_tmp**2)
    #Calculate (BB + theta_11)
    cc = np.arccos((AA_tmp / denom_AA))
    #Calculate BB
    bb1 = np.arccos(bb_tmp / denom_AA)
    bb2 = np.arcsin(cc_tmp/ denom_AA)
    #Calculate theta_11
    theta_11 = cc - bb1
    #Calculate theta_12
    theta_12 = np.arcsin((x1[2] - l_11 * np.sin(theta_11)) / l_12)
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "theta_11": theta_11 * 180 / np.pi,
        "theta_12": theta_12 * 180 / np.pi,
        }

# Example usage
result = calculate_vectors_and_angles_1(10, 10, 10, 10, x1, x2, x3)
print("x1:", result["x1"],"x2:",result["x2"],"x3:",result["x3"])
theta_11 = result["theta_11"]
theta_12 = result["theta_12"]
print("theta_11:", result["theta_11"])
print("theta_12:", result["theta_12"])

def calculate_vectors_and_angles_2(l, l_2, l_21, l_22, x1,x2,x3):
    # Calculating theta_21
    #Calculate numerator in pieces
    y_tmp = np.sqrt(x2[0]**2+x2[1]**2)
    z_tmp = x2[2]
    AA_tmp =  -(l_22**2 - (l_2 - y_tmp)**2 - l_21**2 - z_tmp**2)
    bb_tmp = (2 * (l_2 - y_tmp) * l_21)
    cc_tmp = (2 * z_tmp * l_21)
    #Calculate denominator
    denom_AA = np.sqrt(bb_tmp**2 + cc_tmp**2)
    #Compute BB-theta_21
    cc = np.arcsin(AA_tmp / denom_AA)
    #Compute BB
    bb1 = np.arcsin(bb_tmp / denom_AA)
    bb2 = np.arccos(cc_tmp/ denom_AA)
    #Compute theta_21 and theta_22
    theta_21 = cc + bb1
    theta_22 = np.arcsin((x2[2] - l_21 * np.sin(theta_21)) / l_22)

    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "theta_21": theta_21 * 180 / np.pi,
        "theta_22": theta_22 * 180 / np.pi,
        "bb1": bb1 * 180 / np.pi,
        "bb2": bb2 * 180 / np.pi,
        "cc": cc * 180 / np.pi
    }



result = calculate_vectors_and_angles_2(l,l_2, l_21, l_22,x1,x2,x3)
print("x2:", result["x2"])
theta_21 = result["theta_21"]
theta_22 = result["theta_22"]
print("theta_21:", result["theta_21"])
print("theta_22:", result["theta_22"])

def calculate_vectors_and_angles_3(l, l_3, l_31, l_32, x1,x2,x3):
    # Calculating theta_11
    #Compute the numerator in pieces
    y_tmp = np.sqrt(x3[0]**2+x3[1]**2)
    z_tmp = x3[2]
    AA_tmp = l_32**2 - (l_3 - y_tmp )**2 - l_31**2 - z_tmp**2
    bb_tmp = (2 * (l_3 - y_tmp) * l_31)
    cc_tmp = (2 * z_tmp * l_31)
    #Compute denominator
    denom_AA = np.sqrt(bb_tmp**2 + cc_tmp**2)
    #Compute BB-theta_31
    cc = np.arccos(AA_tmp / denom_AA)
    #Compute BB
    bb1 = np.arccos(bb_tmp / denom_AA)
    bb2 = np.arcsin(cc_tmp/ denom_AA)
    #Compute theta_31 and theta_32
    theta_31 = cc - bb2
    theta_32 = np.arcsin((z_tmp - l_31 * np.sin(theta_31)) / l_32)

    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "theta_31": theta_31 * 180 / np.pi,
        "theta_32": theta_32 * 180 / np.pi,
    }
result = calculate_vectors_and_angles_3(l,l_3, l_31, l_32,x1,x2,x3)
print("x3:", result["x3"])
theta_31 = result["theta_31"]
theta_32 = result["theta_32"]
print("theta_31:", result["theta_31"])
print("theta_32:", result["theta_32"])

def inverse_kinematics(l, l_1, l_11, l_12, l_2, l_21, l_22, l_3, l_31, l_32, nrm,S):
    # Calculate the angles and vectors
    result = position_and_orientation(nrm, S)
    x1 = result["x1"]
    x2 = result["x2"]
    x3 = result["x3"]  
    result = calculate_vectors_and_angles_1(l,l_1,l_11,l_12, x1, x2, x3)
    theta_11 = result["theta_11"]
    theta_12 = result["theta_12"]
    result = calculate_vectors_and_angles_2(l,l_2, l_21, l_22,x1,x2,x3)
    theta_21 = result["theta_21"]
    theta_22 = result["theta_22"]
    result = calculate_vectors_and_angles_3(l,l_3, l_31, l_32,x1,x2,x3)
    theta_31 = result["theta_31"]
    theta_32 = result["theta_32"]
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "theta_11": theta_11,
        "theta_12": theta_12,
        "theta_21": theta_21,
        "theta_22": theta_22,
        "theta_31": theta_31,
        "theta_32": theta_32
    }
print("-----theta_11:-------", theta_11,l, l_1, l_11, l_12,S)
def calculate_position_1(l, l_1, l_11, l_12, theta_11,S):
    # Calculating
    a = l / (2*np.cos(30 * np.pi / 180))
    AA_tmp = l_1 + l_11*np.cos(theta_11*np.pi/180)
    BB_tmp = -S[2] + l_11*np.sin(theta_11*np.pi/180)
    CC_tmp = a*a - AA_tmp**2 - BB_tmp**2 - l_12**2
    denom_AA = np.sqrt(AA_tmp**2 + BB_tmp**2)
    cc = np.arccos(CC_tmp / (denom_AA*2*l_12))
    bb2 = np.arccos(AA_tmp/ denom_AA)
    theta_12 = (np.pi-(cc - bb2))
    print("theta_12",theta_12*180/np.pi,"bb2",bb2*180/np.pi,"cc",cc*180/np.pi,"theta_11",theta_11)
    x1 = np.array([0,l_1+l_11*np.cos(theta_11*np.pi/180)-l_12*np.cos(theta_12),l_11*np.sin(theta_11*np.pi/180)+l_12*np.sin(theta_12)])
    return {
        "x1": x1,
        "theta_11": theta_11,
        "theta_12": theta_12 * 180 / np.pi,
    }
result = calculate_position_1(l,l_1,l_11,l_12,theta_11, S)
x1 = result["x1"]
print("x1:", result["x1"],"theta_11",result["theta_11"],"theta_12",result["theta_12"])

def calculate_position_2(l, l_2, l_21, l_22, theta_21, S, x1):
    # Calculating
    a = l / (2*np.cos(30 * np.pi / 180))
    rth_21 = theta_21*np.pi/180
    AA_tmp =-(l_2 + l_21*np.cos(rth_21))
    BB_tmp = S[2] - l_21*np.sin(rth_21)
    CC_tmp = (a*a - AA_tmp**2 - BB_tmp**2 - l_22**2)
    denom_AA = np.sqrt(AA_tmp**2 + BB_tmp**2)
    cc = np.arcsin(-CC_tmp / (denom_AA*2*l_22))
    bb1 = np.arccos(BB_tmp / denom_AA)
    bb2 = np.arcsin(AA_tmp/ denom_AA)
    theta_22 = (np.pi-(cc-bb2))
    zz = l_21*np.sin(rth_21)+l_22*np.sin(theta_22)
    alpha = np.arcsin((zz-S[2])/a)
    yy=np.sqrt(a*a-(zz-S[2])**2)
    #yy = (l_2+l_21*np.cos(rth_21)-l_22*np.cos(theta_22))#/np.cos(alpha)
    x2 = np.array([np.cos(30*np.pi/180)*yy,-np.sin(30*np.pi/180)*yy,zz])
    yyy = (a*a - (zz-S[2])**2 - l*l + (zz-x1[2])**2 + x1[1]**2)/(2*x1[1])
    print(a*a - (zz - S[2])**2 - yyy**2,yyy)
    xxx = np.sqrt(a*a - (zz - S[2])**2 - yyy**2)
    x2 = np.array([xxx,yyy,zz])
    return {
        "x2": x2,
        "theta_21": theta_21,
        "theta_22": theta_22 * 180 / np.pi,
    }
result = calculate_position_2(l,l_2,l_21,l_22,theta_21,S,x1)
x2 = result["x2"]    
print("x2:", result["x2"],"theta_21",result["theta_21"],"theta_22",result["theta_22"])

def calculate_position_3(l, l_3, l_31, l_32, theta_31,S):
    # Calculating
    a = l / (2*np.cos(30 * np.pi / 180))
    rth_31 = theta_31*np.pi/180
    AA_tmp =  -(l_3  + l_31*np.cos(rth_31))
    BB_tmp = S[2] - l_31*np.sin(rth_31)
    CC_tmp = a*a - AA_tmp**2 - BB_tmp**2 - l_32**2
    denom_AA = np.sqrt(AA_tmp**2 + BB_tmp**2)
    cc = np.arcsin(-CC_tmp / (denom_AA*2*l_32))
    bb1 = np.arccos(BB_tmp / denom_AA)
    bb2 = np.arcsin(AA_tmp/ denom_AA)
    theta_32 = (np.pi-(cc - bb2))
    zz = l_31*np.sin(rth_31)+l_32*np.sin(theta_32)
    #alpha = np.arcsin((zz-S[2])/a)
    yy = (l_3+l_31*np.cos(rth_31)-l_32*np.cos(theta_32))#/np.cos(alpha)
    yy=np.sqrt(a*a-(zz-S[2])**2)
    #print("a", np.sqrt(yy**2+(S[2]-zz)**2))
    x3 = np.array([-np.cos(30*np.pi/180)*yy,-np.sin(30*np.pi/180)*yy,zz])
    return {
        "x3": x3,
        "theta_31": theta_31,
        "theta_32": theta_32 * 180 / np.pi,
    }
result = calculate_position_3(l,l_3,l_31,l_32,theta_31,S)
x3 = result["x3"]     
print("x3:", result["x3"],"theta_31",result["theta_31"],"theta_32",result["theta_32"])
print("mag",np.sqrt((x1[0]-S[0])**2+(x1[1]-S[1])**2+(x1[2]-S[2])**2),np.sqrt((x2[0]-S[0])**2+(x2[1]-S[1])**2+(x2[2]-S[2])**2),np.sqrt((x3[0]-S[0])**2+(x3[1]-S[1])**2+(x3[2]-S[2])**2))
def Forward_Kinematics(l, l_1, l_11, l_12, l_2, l_21, l_22, l_3, l_31, l_32, theta_11, theta_21, theta_31, S):
    # Calculate the angles and vectors
    result = calculate_position_1(l,l_1,l_11,l_12,theta_11,S)
    x1 = result["x1"]
    result = calculate_position_2(l,l_2,l_21,l_22,theta_21,S,x1)
    x2 = result["x2"]
    result = calculate_position_3(l,l_3,l_31,l_32,theta_31,S)
    x3 = result["x3"]
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "theta_11": theta_11,
        "theta_12": theta_12,
        "theta_21": theta_21
    }
result = Forward_Kinematics(l, l_1, l_11, l_12, l_2, l_21, l_22, l_3, l_31, l_32, theta_11, theta_21, theta_31, S)
x1 = result["x1"]
x2 = result["x2"]
x3 = result["x3"]     
# print("x1:", result["x1"],"theta_11",result["theta_11"],"theta_12",result["theta_12"])
# print("x2:", result["x2"],"theta_21",result["theta_21"],"theta_22",result["theta_22"])
# print("x3:", result["x3"],"theta_31",result["theta_31"],"theta_32",result["theta_32"])
# print("mag",np.sqrt((x1[0]-S[0])**2+(x1[1]-S[1])**2),np.sqrt((x2[0]-S[0])**2+(x2[1]-S[1])**2),np.sqrt((x3[0]-S[0])**2+(x3[1]-S[1])**2))
# print("ratio2",x1[0]/x1[1],np.arctan(1/(x2[0]/x2[1])),np.arctan(1/(x3[0]/x3[1])))
# print("croos prod x1 and nrm",np.dot(S-x1,nrm))
# print("croos prod x2 and nrm",np.dot(S-x2,nrm))
# print("croos prod x3 and nrm",np.dot(S-x3,nrm))
# print("l", np.linalg.norm(x1-x2),np.linalg.norm(x2-x3),np.linalg.norm(x3-x1))

def eeq1(d_1,nrm1):
    #print("d_1",d_1)
    n_x = nrm1[0]
    n_y = nrm1[1]
    n_z = nrm1[2]

    A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2



    A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2


    s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
    s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)
    #print("s2",s2,"s3",s3)
    eq1 = (n_x * n_y * ((-BBB + s3) * A2 + A3 * (BB - s2)) * ((-BBB + s3) * A2 - A3 * (BB - s2)) * np.sqrt(3) + 2 * (-4 * l ** 2 * n_z ** 2 * A3 ** 2 + (-BBB + s3) ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) * A2 ** 2 - 2 * (BB - s2) * (n_z ** 2 + 0.3e1 / 0.2e1 * n_x ** 2 - n_y ** 2 / 2) * (-BBB + s3) * A3 * A2 + 2 * (BB - s2) ** 2 * A3 ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) / n_z ** 2 / A2 ** 2 / A3 ** 2 / 8
    #print("eq1",eq1)
    return eq1
def eeq2(d_1,nrm1):
    #print("d_1",d_1)
    n_x = nrm1[0]
    n_y = nrm1[1]
    n_z = nrm1[2]

    A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2



    A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2


    s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
    s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)
    #print("s2",s2,"s3",s3)
    eq2 = (n_x * n_y * ((-BBB + s3) * A2 + A3 * (BB + s2)) * ((-BBB + s3) * A2 - A3 * (BB + s2)) * np.sqrt(3) + 2 * (-4 * l ** 2 * n_z ** 2 * A3 ** 2 + (-BBB + s3) ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) * A2 ** 2 - 2 * (BB + s2) * (n_z ** 2 + 0.3e1 / 0.2e1 * n_x ** 2 - n_y ** 2 / 2) * (-BBB + s3) * A3 * A2 + 2 * (BB + s2) ** 2 * A3 ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) / n_z ** 2 / A2 ** 2 / A3 ** 2 / 8
    #print("eq2",eq2)
    return eq2

def eeq3(d_1,nrm1):
    #print("d_1",d_1)
    n_x = nrm1[0]
    n_y = nrm1[1]
    n_z = nrm1[2]

    A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2



    A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2


    s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
    s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)
    #print("s2",s2,"s3",s3)
    eq3 = (n_x * n_y * ((BBB + s3) * A2 - A3 * (BB - s2)) * ((BBB + s3) * A2 + A3 * (BB - s2)) * np.sqrt(3) + 2 * (-4 * l ** 2 * n_z ** 2 * A3 ** 2 + (BBB + s3) ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) * A2 ** 2 + 2 * (BB - s2) * (n_z ** 2 + 0.3e1 / 0.2e1 * n_x ** 2 - n_y ** 2 / 2) * A3 * (BBB + s3) * A2 + 2 * (BB - s2) ** 2 * A3 ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) / n_z ** 2 / A2 ** 2 / A3 ** 2 / 8
    #print("eq3",eq3)
    return eq3
def eeq4(d_1, nrm1):
    #print("d_1",d_1)
    n_x = nrm1[0]
    n_y = nrm1[1]
    n_z = nrm1[2]

    A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2



    A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2


    s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
    s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)
    #print("s2",s2,"s3",s3)
    eq4 = (n_x * n_y * ((BBB + s3) * A2 - A3 * (BB + s2)) * ((BBB + s3) * A2 + A3 * (BB + s2)) * np.sqrt(3) + 2 * (-4 * l ** 2 * n_z ** 2 * A3 ** 2 + (BBB + s3) ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) * A2 ** 2 + 2 * (BB + s2) * (n_z ** 2 + 0.3e1 / 0.2e1 * n_x ** 2 - n_y ** 2 / 2) * A3 * (BBB + s3) * A2 + 2 * (BB + s2) ** 2 * A3 ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) / n_z ** 2 / A2 ** 2 / A3 ** 2 / 8
    #print("eq4",eq4)
    return eq4

# Initial guess for the root

def triangle_orientation_and_location(nrm1,S,initial_guess):
    n_x = nrm1[0]
    n_y = nrm1[1]
    n_z = nrm1[2]
    root = fsolve(eeq1, initial_guess,nrm1)
    #print("ang1",ang1,"ang2",ang2)
    print("n_x {:.5f}".format(n_x),"n_y {:.5f}".format(n_y),"n_z {:.5f}".format(n_z),"{:.5f}".format(root[0]))
    
    d_1 = root[0]
    A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2

    A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l ** 2

    s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
    s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)

    d_2 = (-BB+s2)/(2*A2)
    d_3 = (-BBB+s3)/(2*A3)
    c_1 = 12
    c_2 = c_1 + (1/n_z)*(-d_2*np.cos(30*np.pi/180)*n_x + d_2*np.sin(30*np.pi/180)*n_y + d_1*n_y)
    #c_3 = c_1 + (1/n_z)*( d_3*np.cos(30*np.pi/180)*n_x + d_2*np.sin(30*np.pi/180)*n_y + d_1*n_y)
    c_3 = c_2 + (1/n_z)*( (d_2+d_3)*np.cos(30*np.pi/180)*n_x -(-d_3+d_2)*np.sin(30*np.pi/180)*n_y)
    #print(c_3)

    X1_hat = np.array([0, 1, 0])
    X2_hat = np.array([np.cos(30*np.pi/180), -np.sin(30*np.pi/180), 0])
    X3_hat = np.array([-np.cos(30*np.pi/180), -np.sin(30*np.pi/180), 0])
    z_hat =  np.array([0, 0, 1])

    P1 = d_1*X1_hat + c_1*z_hat
    P2 = d_2*X2_hat + c_2*z_hat
    P3 = d_3*X3_hat + c_3*z_hat

    pp = (P1+P2+P3)/3 - S

    P1 = P1 - pp
    P2 = P2 - pp
    P3 = P3 - pp
    result = calculate_vectors_and_angles_1(l,l_1,l_11,l_12, P1, P2, P3)
    theta_11 = result["theta_11"]
    theta_12 = result["theta_12"]
    result = calculate_vectors_and_angles_2(l,l_2, l_21, l_22,P1, P2, P3)
    theta_21 = result["theta_21"]
    theta_22 = result["theta_22"]
    result = calculate_vectors_and_angles_3(l,l_3, l_31, l_32,P1, P2, P3)
    theta_31 = result["theta_31"]
    theta_32 = result["theta_32"]
    return{
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "theta_11": theta_11,
        "theta_12": theta_12,
        "theta_21": theta_21,
        "theta_22": theta_22,
        "theta_31": theta_31,
        "theta_32": theta_32
    }

print("========================Begin For loop==========================")
for ang1 in range(0,91,45):
    initial_guess = 5.0
    alpha=ang1*np.pi/180
    for ang2 in range(80,89,15):
        print("ang1 := ",ang1,"ang2 := ", ang2)

        phi=ang2*np.pi/180
        def nn_x():
            return np.cos(phi)*np.cos(alpha )
        def nn_y():
            return np.cos(phi)*np.sin(alpha )
        def nn_z():
            return np.sin(phi)
        n_x = nn_x()
        n_y = nn_y()
        n_z = nn_z()        # Find the root using fsolve
       
        S = np.array([0,0,12])

        nrm[0] = n_x
        nrm[1] = n_y
        nrm[2] = n_z 

        result = triangle_orientation_and_location(nrm,S,initial_guess)
        x1 = result["P1"]
        x2 = result["P2"]
        x3 = result["P3"]  
        theta_11 = result["theta_11"]
        theta_12 = result["theta_12"]
        theta_31 = result["theta_21"]
        theta_32 = result["theta_22"]
        theta_21 = result["theta_31"]
        theta_22 = result["theta_32"]

        print("n_x {:.5f}".format(n_x),"n_y {:.5f}".format(n_y),"n_z {:.5f}".format(n_z),"dot {:.5f}".format(np.dot(nrm,x2-x1)),"dot {:.5f}".format(np.dot(nrm,x3-x1)),"dot {:.5f}".format(np.dot(nrm,x3-x2)),"norm {:.5f}"
              .format(np.linalg.norm(x2-x1)),"norm {:.5f}".format(np.linalg.norm(x3-x1)),"norm {:.5f}".format(np.linalg.norm(x2-x3)))
        print("theta_11", theta_11,"theta_12", theta_12,"theta_21", theta_21,"theta_22", theta_22,"theta_31", theta_31,"theta_32", theta_32)
        print("S",S,"x1",x1,"x2",x2,"x3",x3,"cent",(1/3)*(x1+x2+x3),"nrm",nrm)


# Create a 3D figure

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the circle
theta = np.linspace(0, 2 * np.pi, 100)
radius = l_1
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)
z_circle = np.zeros_like(theta)  # Circle lies in the XY plane

# Plot the circle
ax.plot(x_circle, y_circle, z_circle, color='red', label='Circle')

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Set axis limits
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(0, 30)



# Define the vertices of the triangle
triangle_vertices1 = np.array([
    x1,  # Vertex 1
    x2,  # Vertex 2
    x3,  # Vertex 3
    x1   # Close the triangle
])
# Plot the triangle
#ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], color='blue', label='Triangle')
# Define the vertices of the triangle
triangle_vertices2 = np.array([
    [0, l_1, 0],  # Vertex 1
    [0, l_11*np.cos(theta_11*np.pi/180)+l_1, l_11*np.sin(theta_11*np.pi/180)],  # Vertex 2
    [0, l_11*np.cos(theta_11*np.pi/180)+l_1-l_12*np.cos(theta_12*np.pi/180), l_11*np.sin(theta_11*np.pi/180)+l_12*np.sin(theta_12*np.pi/180)],  # Vertex 3
    [0, l_1, 0]   # Close the triangle
])

# Plot the triangle
#ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], color='blue', label='Triangle')
# Define the vertices of the triangle
triangle_vertices3 = np.array([
    [-0.866*l_2, -0.5*l_2, 0],  # Vertex 1
    [-0.866*(l_21*np.cos(theta_21*np.pi/180)+l_2), -0.5*(l_21*np.cos(theta_21*np.pi/180)+l_2), l_21*np.sin(theta_21*np.pi/180)],  # Vertex 2
    [-0.866*(l_21*np.cos(theta_21*np.pi/180)+l_2-l_22*np.cos(theta_22*np.pi/180)), -0.5*(l_21*np.cos(theta_21*np.pi/180)+l_2-l_22*np.cos(theta_22*np.pi/180)), l_21*np.sin(theta_21*np.pi/180)+l_22*np.sin(theta_22*np.pi/180)],  # Vertex 3
    [-0.866*l_2, -0.5*l_2, 0]  # Close the triangle
])

# Plot the triangle
#ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], color='blue', label='Triangle')
# Define the vertices of the triangle
triangle_vertices4 = np.array([
    [0.866*l_3, -0.5*l_3, 0],  # Vertex 1
    [0.866*(l_31*np.cos(theta_31*np.pi/180)+l_3), -0.5*(l_31*np.cos(theta_31*np.pi/180)+l_3), l_31*np.sin(theta_31*np.pi/180)],  # Vertex 2
    [0.866*(l_31*np.cos(theta_31*np.pi/180)+l_3-l_32*np.cos(theta_32*np.pi/180)), -0.5*(l_31*np.cos(theta_31*np.pi/180)+l_3-l_32*np.cos(theta_32*np.pi/180)), l_31*np.sin(theta_31*np.pi/180)+l_32*np.sin(theta_32*np.pi/180)],  # Vertex 3
    [0.866*l_3, -0.5*l_3, 0]  # Close the triangle
])

# Plot the triangle
#ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], color='blue', label='Triangle')
# Plot the initial triangle
line1, = ax.plot(triangle_vertices1[:, 0], triangle_vertices1[:, 1], triangle_vertices1[:, 2], color='blue')

# For triangles 2, 3, 4 - create temporary initial lines (will be updated in animation)
# Triangle 2 polylines (1→2 and 2→3)
line2_seg1, = ax.plot([0, 0], [l_1, l_1], [0, 0], color='red')  # Initial dummy line
line2_seg2, = ax.plot([0, 0], [l_1, l_1], [0, 0], color='red')  # Initial dummy line

# Triangle 3 polylines (1→2 and 2→3)  
line3_seg1, = ax.plot([0.866*l_2, 0.866*l_2], [-0.5*l_2, -0.5*l_2], [0, 0], color='green')  # Initial dummy line
line3_seg2, = ax.plot([0.866*l_2, 0.866*l_2], [-0.5*l_2, -0.5*l_2], [0, 0], color='green')  # Initial dummy line

# Triangle 4 polylines (1→2 and 2→3)
line4_seg1, = ax.plot([-0.866*l_3, -0.866*l_3], [-0.5*l_3, -0.5*l_3], [0, 0], color='yellow')  # Initial dummy line
line4_seg2, = ax.plot([-0.866*l_3, -0.866*l_3], [-0.5*l_3, -0.5*l_3], [0, 0], color='yellow')  # Initial dummy line

# Animation function
def update(frame):
    # Rotate the triangle around the Z-axis
    alpha = np.deg2rad(frame*12 % 360)
    phi = 75* np.pi / 180
    #print("frame",frame,"alpha",alpha,"phi",phi)
    nrm = np.array([np.cos(phi)*np.cos(alpha), np.cos(phi)*np.sin(alpha), np.sin(phi)])

    initial_guess = 5.0
      
    S = np.array([0,0,12])
    result = triangle_orientation_and_location(nrm,S,initial_guess)
    x1 = result["P1"]
    x2 = result["P2"]
    x3 = result["P3"]  
    theta_11 = result["theta_11"]
    theta_12 = result["theta_12"]
    theta_21 = result["theta_21"]
    theta_22 = result["theta_22"]
    theta_31 = result["theta_31"]
    theta_32 = result["theta_32"]
    # result = inverse_kinematics(l, l_1, l_11, l_12, l_2, l_21, l_22, l_3, l_31, l_32, nrm, S)
    # fx1 = result["x1"]
    # fx2 = result["x2"]
    # fx3 = result["x3"]
    # theta_11 = result["theta_11"]
    # theta_12 = result["theta_12"]
    # theta_21 = result["theta_21"]
    # theta_22 = result["theta_22"]
    # theta_31 = result["theta_31"]
    # theta_32 = result["theta_32"]
    # result = Forward_Kinematics(l, l_1, l_11, l_12, l_2, l_21, l_22, l_3, l_31, l_32, theta_11, theta_21, theta_31, S)
    # x1 = result["x1"]
    # x2 = result["x2"]
    # x3 = result["x3"]
    if frame % 100 == 0:
        print("frame",frame,"alpha",alpha,"phi",phi)
        # print("----x1:", x1)
        # print("x2:", x2)
        # print("x3:", x3)
        # S = (1/3)*(x1+x2+x3)
        print("mag",np.sqrt((x1[0]-x2[0])**2++(x1[1]-x2[1])**2+(x1[2]-x2[2])**2),np.sqrt((x2[0]-x3[0])**2+(x2[1]-x3[1])**2+(x2[2]-x3[2])**2),np.sqrt((x3[0]-x1[0])**2+(x3[1]-x1[1])**2+(x3[2]-x1[2])**2))
    # Define the vertices of the triangle
    triangle_vertices1 = np.array([
        x1,  # Vertex 1
        x2,  # Vertex 2
        x3,  # Vertex 3
        x1   # Close the triangle
    ])

    # Create polylines for triangles 2, 3, 4 (only connecting 1→2 and 2→3)
    # Triangle 2 vertices
    vertex2_1 = np.array([0, l_1, 0])
    vertex2_2 = np.array([0, l_11*np.cos(theta_11*np.pi/180)+l_1, l_11*np.sin(theta_11*np.pi/180)])
    vertex2_3 = np.array([0, l_11*np.cos(theta_11*np.pi/180)+l_1-l_12*np.cos(theta_12*np.pi/180), l_11*np.sin(theta_11*np.pi/180)+l_12*np.sin(theta_12*np.pi/180)])
    
    triangle_vertices2_line1 = np.array([vertex2_1, vertex2_2])  # Line from point 1 to 2
    triangle_vertices2_line2 = np.array([vertex2_2, vertex2_3])  # Line from point 2 to 3

    # Triangle 3 vertices
    vertex3_1 = np.array([0.866*l_2, -0.5*l_2, 0])
    vertex3_2 = np.array([0.866*(l_21*np.cos(theta_21*np.pi/180)+l_2), -0.5*(l_21*np.cos(theta_21*np.pi/180)+l_2), l_21*np.sin(theta_21*np.pi/180)])
    vertex3_3 = np.array([0.866*(l_21*np.cos(theta_21*np.pi/180)+l_2-l_22*np.cos(theta_22*np.pi/180)), -0.5*(l_21*np.cos(theta_21*np.pi/180)+l_2-l_22*np.cos(theta_22*np.pi/180)), l_21*np.sin(theta_21*np.pi/180)+l_22*np.sin(theta_22*np.pi/180)])
    
    triangle_vertices3_line1 = np.array([vertex3_1, vertex3_2])  # Line from point 1 to 2
    triangle_vertices3_line2 = np.array([vertex3_2, vertex3_3])  # Line from point 2 to 3

    # Triangle 4 vertices
    vertex4_1 = np.array([-0.866*l_3, -0.5*l_3, 0])
    vertex4_2 = np.array([-0.866*(l_31*np.cos(theta_31*np.pi/180)+l_3), -0.5*(l_31*np.cos(theta_31*np.pi/180)+l_3), l_31*np.sin(theta_31*np.pi/180)])
    vertex4_3 = np.array([-0.866*(l_31*np.cos(theta_31*np.pi/180)+l_3-l_32*np.cos(theta_32*np.pi/180)), -0.5*(l_31*np.cos(theta_31*np.pi/180)+l_3-l_32*np.cos(theta_32*np.pi/180)), l_31*np.sin(theta_31*np.pi/180)+l_32*np.sin(theta_32*np.pi/180)])
    
    triangle_vertices4_line1 = np.array([vertex4_1, vertex4_2])  # Line from point 1 to 2
    triangle_vertices4_line2 = np.array([vertex4_2, vertex4_3])  # Line from point 2 to 3

    line1.set_data(triangle_vertices1[:, 0], triangle_vertices1[:, 1])
    line1.set_3d_properties(triangle_vertices1[:, 2])

    # Update triangle 2 polylines (1→2 and 2→3)
    line2_seg1.set_data(triangle_vertices2_line1[:, 0], triangle_vertices2_line1[:, 1])
    line2_seg1.set_3d_properties(triangle_vertices2_line1[:, 2])
    line2_seg2.set_data(triangle_vertices2_line2[:, 0], triangle_vertices2_line2[:, 1])
    line2_seg2.set_3d_properties(triangle_vertices2_line2[:, 2])

    # Update triangle 3 polylines (1→2 and 2→3)
    line3_seg1.set_data(triangle_vertices3_line1[:, 0], triangle_vertices3_line1[:, 1])
    line3_seg1.set_3d_properties(triangle_vertices3_line1[:, 2])
    line3_seg2.set_data(triangle_vertices3_line2[:, 0], triangle_vertices3_line2[:, 1])
    line3_seg2.set_3d_properties(triangle_vertices3_line2[:, 2])

    # Update triangle 4 polylines (1→2 and 2→3)
    line4_seg1.set_data(triangle_vertices4_line1[:, 0], triangle_vertices4_line1[:, 1])
    line4_seg1.set_3d_properties(triangle_vertices4_line1[:, 2])
    line4_seg2.set_data(triangle_vertices4_line2[:, 0], triangle_vertices4_line2[:, 1])
    line4_seg2.set_3d_properties(triangle_vertices4_line2[:, 2])

    return line1, line2_seg1, line2_seg2, line3_seg1, line3_seg2, line4_seg1, line4_seg2

# Create the animation
ani = FuncAnimation(fig, update, frames=1000, interval=100, blit=False)

# Option 1: Save specific frames as static images
# Uncomment the lines below to save frames instead of showing animation
# for frame_num in [0, 100, 200, 300, 400, 500]:  # Save frames at different positions
#     update(frame_num)  # Update the plot to the specific frame
#     plt.savefig(f'stewart_platform_frame_{frame_num}.png', dpi=300, bbox_inches='tight')
#     plt.savefig(f'stewart_platform_frame_{frame_num}.pdf', bbox_inches='tight')  # PDF for LaTeX

# Option 2: Save the animation as GIF or MP4
# Uncomment ONE of the lines below to save the animation
# ani.save('stewart_platform_animation.gif', writer='pillow', fps=10)
# ani.save('stewart_platform_animation.mp4', writer='ffmpeg', fps=10)

# Show the plot
plt.show() 