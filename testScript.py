# file สำหรับตรวจคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส
1.อภิเชษฐ์_6560
2.ณัฐพงศ์_6570

'''


import numpy as np
from HW3_utils import FKHW3
import roboticstoolbox as rtb
from spatialmath import SE3
#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 1>====================================================#
#code here


q = [0, 0, 0]
print(f"Check Jacobian : ")
print(f"Input : {q}")
def endEffectorJacobianHW3(q):
    R, P, R_e, p_e = FKHW3(q)
    # R = เมทริกซ์การหมุนของแต่ละเฟรม, P = ตำแหน่งของแต่ละเฟรม
    # R_e = เมทริกซ์การหมุนของ End Effector, p_e = ตำแหน่งของ End Effector
    
    # สร้างเมทริกซ์ J_v และ J_w ขนาด 3x3 เพื่อเก็บผลลัพธ์จากการคำนวณ Jacobian
    J_v = np.zeros((3, 3))  # สำหรับส่วนความเร็วเชิงเส้น (Linear velocity)
    J_w = np.zeros((3, 3))  # สำหรับส่วนความเร็วเชิงมุม (Angular velocity)

    # วนลูปคำนวณ Jacobian สำหรับแต่ละข้อต่อ (มี 3 ข้อ)
    for i in range(3):
        # ดึงเมทริกซ์การหมุน (R_i) และตำแหน่ง (P_i) ของข้อต่อที่ i
        R_i = R[:, :, i]
        P_i = P[:, i]
        
        # คำนวณแกนหมุน z_i ของข้อต่อที่ i (ในเฟรมของข้อต่อที่ i)
        z = R_i @ np.array([0, 0, 1])  # แกนหมุนของแต่ละข้อต่อ (ใน local frame)

        # คำนวณส่วนของความเร็วเชิงเส้น (J_v) โดยการทำ cross product ระหว่างแกนหมุนและระยะห่างจากตำแหน่งข้อต่อถึง End Effector
        J_v[:, i] = np.cross(z, (p_e - P_i))  # ผลลัพธ์เป็นเวกเตอร์ 3x1
        
        # คำนวณส่วนของความเร็วเชิงมุม (J_w) สำหรับแต่ละข้อต่อ โดยใช้แกนหมุน z_i ที่คำนวณได้
        J_w[:, i] = z

    # รวมส่วนของความเร็วเชิงเส้นและความเร็วเชิงมุมเข้าด้วยกันเป็น Jacobian ขนาด 6x3
    J_e = np.vstack((J_v, J_w))  # นำ J_v และ J_w มาต่อกันในแนวตั้ง
    return J_e  # คืนค่า Jacobian
# เรียกใช้ฟังก์ชัน endEffectorJacobianHW3 เพื่อคำนวณค่า Jacobian
J_ours = endEffectorJacobianHW3(q)
# แสดงผลค่า Jacobian ที่คำนวณได้

print("Our Jacobian Matrix :")
print(endEffectorJacobianHW3(q))


#ตัวเเปรค่าต่างๆ
d_1 = 0.0892
a_2 = -0.425
a_3 = -0.39243
d_4 = 0.109
d_5 = 0.093
d_6 = 0.082
# สร้างโมเดลหุ่นยนต์โดยใช้ Denavit-Hartenberg parameters และกำหนด Tool Frame
robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=d_1, offset=np.pi),   # ข้อต่อที่ 1
        rtb.RevoluteMDH(alpha=np.pi/2),         # ข้อต่อที่ 2
        rtb.RevoluteMDH(a=a_2)                  # ข้อต่อที่ 3
    ],
    tool=  SE3.Tx(a_3) @SE3.Tz(d_4) @ SE3.Rx(np.pi/2) @SE3.Tz(d_5) @SE3.Rz(-np.pi/2) @ SE3.Rx(-np.pi/2) @SE3.Tz(-d_6),
    name="RRR_Robot"
)
# คำนวณ Jacobian ที่ตำแหน่งข้อต่อ [0, 0, 0]
q = [0, 0, 0]  # ค่าเริ่มต้นของข้อต่อ
J = robot.jacob0(q)  # คำนวณ Jacobian matrix จาก Robotic Toolbox
J_rtb = robot.jacob0(q)  # คำนวณอีกรอบสำหรับการเปรียบเทียบ
print("Jacobian Matrix form roboticstoolbox :")
print(J)  # แสดง Jacobian matrix
# เปรียบเทียบค่า Jacobian ที่คำนวณเองกับค่า Jacobian จาก Robotic Toolbox
is_close = np.allclose(J_ours, J_rtb, atol=1e-6)  # ตรวจสอบว่าค่าทั้งสองใกล้เคียงกันหรือไม่
# แสดงผลลัพธ์ว่าค่าที่คำนวณออกมาจากสองวิธีการใกล้เคียงกันหรือไม่
print(f"Are the Jacobians close? {is_close}")
print("-" * 50) 



#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 2>====================================================#
#code here
def checkSingularityHW3(q):
    epsilon = 0.001
    # คำนวณ Jacobian Matrix ของเราเอง
    J_e1 = endEffectorJacobianHW3(q)
    # หาค่า determinant ของส่วน linear velocity (3x3 matrix)
    J_v1 = J_e1[:3, :3]
    det_J1 = np.linalg.det(J_v1)
    # ตรวจสอบว่าอยู่ใน Singular State หรือไม่ (ถ้า determinant ใกล้ 0 ถือว่าอยู่ใน Singular State)
    if abs(det_J1) < epsilon:
        flag = 1  # Singular
    else:
        flag = 0  # Non-Singular
    return flag

# ฟังก์ชันนี้ใช้ Robotics Toolbox เพื่อตรวจสอบ Singular
def checkSingularityWithToolbox(q):
    epsilon = 0.001
    
    # คำนวณ Jacobian matrix จาก Robotics Toolbox
    J = robot.jacob0(q)
    
    # หาค่า determinant ของส่วน linear velocity (3x3 matrix)
    J_v = J[:3, :3]
    det_J = np.linalg.det(J_v)
    
    # ตรวจสอบ Singular เหมือนกับฟังก์ชันของเรา
    if abs(det_J) < epsilon:
        flag = 1  # Singular
    else:
        flag = 0  # Non-Singular
    return flag

# ฟังก์ชันนี้จะสุ่มค่า q และตรวจสอบ singularity ด้วยทั้งสองฟังก์ชัน จากนั้นเปรียบเทียบผลลัพธ์
def random_test_cases_and_compare(num_tests=5):
    results = []
    for _ in range(num_tests):
        # สุ่มค่า q ระหว่าง -pi ถึง pi สำหรับข้อต่อแต่ละตัว
        q_random = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        
        # ตรวจสอบ Singular ด้วยฟังก์ชันของเราและ Robotics Toolbox
        is_singular_hw3 = checkSingularityHW3(q_random)
        is_singular_toolbox = checkSingularityWithToolbox(q_random)
        
        # ตรวจสอบว่าผลลัพธ์จากทั้งสองฟังก์ชันตรงกันหรือไม่
        match = (is_singular_hw3 == is_singular_toolbox)
        results.append((q_random, is_singular_hw3, is_singular_toolbox, match))
    
    return results

# สุ่มทดสอบ 5 ชุด และแสดงผลลัพธ์
test_results = random_test_cases_and_compare(num_tests=1)
# แสดงผลลัพธ์ว่าค่าที่คำนวณได้จากทั้งสองฟังก์ชันใกล้เคียงกันหรือไม่
print("Check Singularity :")
for idx, (q_random, singular_hw3, singular_toolbox, match) in enumerate(test_results):
    print(f"TestSingularity {idx+1}")
    print(f"Input : q = {q_random}")
    print(f"Output :")

    print(f"   Our Singularity = {singular_hw3} , Robotics Toolbox Singularity = {singular_toolbox} , Check Two Singularity = {match}")
    print("-" * 50) 

#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 3>====================================================#
#code here

def computeEffortHW3(q, w):
    # คำนวณ Jacobian matrix ด้วยวิธีที่คำนวณเองและจาก Robotics Toolbox
    J_e1 = endEffectorJacobianHW3(q)  # Jacobian ที่เราคำนวณเอง
    J_e2 = robot.jacob0(q)  # Jacobian จาก Robotics Toolbox
    # คำนวณ torque (effort) โดยการคูณ Jacobian กับ force/moment vector
    tau1 = J_e1.T @ w  # คำนวณ effort ด้วย Jacobian ที่เราคำนวณเอง
    tau2 = J_e2.T @ w  # คำนวณ effort ด้วย Robotics Toolbox
    return tau1, tau2  # คืนค่า torque จากทั้งสองวิธี

# ฟังก์ชันนี้สุ่มทดสอบ test cases
def random_test_cases(num_tests=10):
    results = []
    for _ in range(num_tests):
        # สุ่มค่า q จากช่วง -pi ถึง pi สำหรับแต่ละข้อต่อ
        q_random = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        
        # สุ่มค่า w (force/moment vector) จากช่วง -100 ถึง 100
        w_random = np.random.uniform(low=-100, high=100, size=6)
        
        # คำนวณ effort (torque) ด้วยสองวิธี (เราคำนวณเอง และจาก Robotics Toolbox)
        tau1, tau2 = computeEffortHW3(q_random, w_random)
        
        # ตรวจสอบว่าผลลัพธ์ของ torque จากทั้งสองวิธีใกล้เคียงกันหรือไม่
        is_close = np.allclose(tau1, tau2, atol=1e-6)
        
        # เก็บผลลัพธ์ที่สุ่มได้และสถานะการเปรียบเทียบ
        results.append((q_random, w_random, tau1, tau2, is_close))
    
    return results  # คืนค่าผลลัพธ์ทั้งหมดที่ได้จากการทดสอบ

# เรียกฟังก์ชันสุ่มทดสอบ 5 test cases
test_results = random_test_cases(num_tests=1)
print("Check Effort :")
# แสดงผลลัพธ์ที่ได้จากการทดสอบ
for idx, (q_random, w_random, tau1, tau2, is_close) in enumerate(test_results):
    print(f"TestEffort {idx+1}")
    print(f"input q = {q_random}")  # แสดงค่า q ที่สุ่มได้
    print(f"input w = {w_random}")  # แสดงค่า w ที่สุ่มได้
    print(f"Effort from Our Method: {tau1}")  # แสดงค่า torque ที่คำนวณด้วยวิธีของเรา
    print(f"Effort from Robotics Toolbox: {tau2}")  # แสดงค่า torque ที่คำนวณด้วย Robotics Toolbox
    print(f"Results are close: {is_close}")  # แสดงว่าผลลัพธ์ทั้งสองใกล้เคียงกันหรือไม่
    print("-" * 50)  # ขีดเส้นแบ่งระหว่างแต่ละทดสอบ

#==============================================================================================================#