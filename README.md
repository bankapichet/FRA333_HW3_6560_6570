# Kinematics Homework 3
-   **FRA333_HW3_60_70.py**: เป็นไฟล์ที่แสดงวิธีการหา  Jacobian ด้วยวิธีที่เราหามา , มีการแสดงวิธีการเช็ค singularities และการคำนวณหา  torques
    
-   **testScript.py**: เป็นการแสดงการเช็คคำตอบวิธีที่เราทำเองกับคำตอบจาก  roboticstoolbox

  

## ขั้นตอนการติดตั้ง

  

### Install Numpy 

```sh
pip3  install  numpy==1.23.5
```

### Install Robotics Toolbox

```sh
pip3  install  roboticstoolbox-python
```

  

### Install Spatialmath

```sh
pip  install  spatialmath
```
ดาวน์โหลดโฟลเดอร์ zip นี้ แตกไฟล์และเปิดโฟลเดอร์ด้วย VScode


## ขั้นตอนการใช้งาน


### Run FRA333_HW3_6560_6570.py 

```sh
python3  .\FRA333_HW3_6560_6570.py
```
---
###  คำถามข้อที่ 1 : หา Jacobian ของหุ่นยนต์
 Jacobian matrix (𝐽) ถูกคำนวณโดยใช้วิธีการหา (Forward kinematics) ซึ่งประกอบด้วย ความเร็วเชิงเส้น (𝐽𝑣) และ ความเร็วเชิงมุม (𝐽𝑤)
 
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian1.png"  />
</p>

<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian2.png"  />
</p>

### Example of use:
สูตรการคำนวณความเร็วเชิงเส้น
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian3.png"  />
</p>

<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian4.png"  />
</p>

สูตรการคำนวณความเร็วเชิงมุม
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian5.png"  />
</p>

<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian6.png"  />
</p>

### Method:
1.  **คำนวณตำแหน่งและการหมุน**
	-   เรียกใช้ฟังก์ชัน Forward Kinematics (`FKHW3(q)`) เพื่อคำนวณ
		 -   `R`: เมทริกซ์การหมุนสำหรับแต่ละข้อต่อ
		-   `P`: ตำแหน่งของข้อต่อ
		-   `R_e`: เมทริกซ์การหมุนของ End Effector
		-   `p_e`: ตำแหน่งของ End Effector
2.  **การคำนวณ Jacobian สำหรับข้อต่อแต่ละข้อ**
   	-    วนลูปคำนวณ Jacobian โดยคำนวณแกนหมุน `z` จากการคูณเมทริกซ์การหมุน `R_i` กับเวกเตอร์ [0, 0, 1] จากนั้นหาความเร็วเชิงเส้น `(J_v)` ด้วยการทำ cross product ระหว่าง `z` กับระยะห่างระหว่างตำแหน่ง End Effector `(p_e)` และตำแหน่งข้อต่อ `(P_i)` ส่วนความเร็วเชิงมุม `(J_w)` คือแกนหมุน `z`
3.  **รวมความเร็วเชิงเส้นและเชิงมุมเข้าด้วยกัน**
	  -  นำเมทริกซ์ `J_v` (ความเร็วเชิงเส้น) และ `J_w` (ความเร็วเชิงมุม) มาต่อกันในแนวตั้งเพื่อสร้าง Jacobian Matrix (`J_e`) ขนาด 6x3

### Example of use:
```bash
J_e = endEffectorJacobianHW3(q)
```

---
###  คำถามข้อที่ 2 : หาสภาวะ Singularity
การตรวจการเข้าสู่สภาวะ (Singularity) ของหุ่นยนต์ โดยเมื่อ determinant ของ Jacobian ส่วนความเร็วเชิงเส้น (Jv​) มีค่าเท่ากับศูนย์ จะบ่งชี้ว่าหุ่นยนต์อยู่ในสภาวะเอกพจน์ (singular configuration) ซึ่งหมายความว่าหุ่นยนต์สูญเสียความสามารถในการเคลื่อนที่ได้อย่างอิสระในบางทิศทาง

**Formula used in code:**
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/det1.png"  />
</p>

การที่ det⁡(Jv) = 0 บ่งบอกถึงสภาวะที่หุ่นยนต์เกิด (Singularity) ซึ่งหมายความว่าหุ่นยนต์ไม่สามารถเคลื่อนที่ได้อย่างอิสระในทุกทิศทาง 

### Method:
1.  **คำนวณเมทริกซ์ Jacobian**
    -   นำค่า Jacobian matrix จากการหาในข้อแรก `J_e1 = endEffectorJacobianHW3(q)` แยกส่วนของความเร็วเชิงเส้น (Linear velocity Jacobian) เพื่อนำมาคำนวณค่า determinant โดยใช้ฟังก์ชัน `np.linalg.det`
2.  **การตรวจสอบค่า Determinant เพื่อหา Singularitie**
    -   หากค่า determinant ของความเร็วเชิงเส้น น้อยกว่า `epsilon` ที่กำหนด (0.001) หุ่นยนต์จะถือว่าอยู่ในสถานะ Singular (flag = 1) มิฉะนั้นจะไม่อยู่ในสถานะ Singular (flag = 0) 

### Example of use:
```bash
singularity = checkSingularityHW3(q)
```

---
#### คำถามข้อที่ 3 : หา effort ของแต่ละข้อต่อเมื่อมี wrench มากระทำ
การคำนวณแรงบิดที่ข้อต่อของหุ่นยนต์  `τ` เพื่อรับแรงภายนอกที่มากระทำที่ end effector`w` โดยใช้ Jacobian Transpose `J_T`ซึ่งบ่งบอกถึงผลกระทบของการเคลื่อนที่ของข้อต่อที่มีต่อตำแหน่งและแรงที่ปลายทาง

### Formula used in code:
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian7.png"  />
</p>

<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian8.png"  />
</p>

### Method:
1.  **คำนวณ Torque**
    -   นำค่า Jacobian ที่คำนวนจากข้อแรก `J_e = endEffectorJacobianHW3(q)`มาทำการ Transpose ได้ Jacobian Transpose (`J_e.T`) และนำมาคูณกับแรงภายนอก `w` ที่กระทำต่อ end effector เพื่อหาค่า torque ที่ข้อต่อต้องออกแรง

### Example of use:
```bash
tua = computeEffortHW3(q,w)
```
