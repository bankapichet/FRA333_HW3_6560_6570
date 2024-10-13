
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
### คำถามข้อที่ 1 : หา Jacobian  Matrix ของหุ่นยนต์

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
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobian9.png"  />
</p>


สูตรการคำนวณความเร็วเชิงมุม
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/jacobain10.png"  />
</p>



### Method:
1.  **คำนวณตำแหน่งและการหมุน**
 เรียกใช้ฟังก์ชัน Forward Kinematics (`FKHW3(q)`) เพื่อคำนวณ
	-   `R`: เมทริกซ์การหมุนสำหรับแต่ละข้อต่อ
	-   `P`: ตำแหน่งของข้อต่อ
	-   `R_e`: เมทริกซ์การหมุนของ End Effector
	-   `p_e`: ตำแหน่งของ End Effector
2.  **การคำนวณ Jacobian สำหรับข้อต่อแต่ละข้อ**
วนลูปคำนวณ Jacobian โดยคำนวณแกนหมุน `z` จากการคูณเมทริกซ์การหมุน `R_i` กับเวกเตอร์ [0, 0, 1] จากนั้นหาความเร็วเชิงเส้น `(J_v)` ด้วยการทำ cross product ระหว่าง `z` กับระยะห่างระหว่างตำแหน่ง End Effector `(p_e)` และตำแหน่งข้อต่อ `(P_i)` ส่วนความเร็วเชิงมุม `(J_w)` คือแกนหมุน `z`
3.  **รวมความเร็วเชิงเส้นและเชิงมุมเข้าด้วยกัน**
นำเมทริกซ์ `J_v` (ความเร็วเชิงเส้น) และ `J_w` (ความเร็วเชิงมุม) มาต่อกันในแนวตั้งเพื่อสร้าง Jacobian Matrix (`J_e`) ขนาด 6x3
### ฟังก์ชั่นในการหา Jacobian ของหุ่นยนต์ :
```bash
def endEffectorJacobianHW3(q: list[float]) -> list[float]:
	# เรียกฟังก์ชันคำนวณตำแหน่งและการหมุนของข้อต่อต่าง ๆ
	R, P, R_e, p_e  =  FKHW3(q)
	# สร้างเมทริกซ์ 3x3 ไว้เก็บค่าความเร็วเชิงเส้น (J_v) และความเร็วเชิงมุม (J_w)
	J_v  =  np.zeros((3, 3)) # ส่วนของความเร็วเชิงเส้น
	J_w  =  np.zeros((3, 3)) # ส่วนของความเร็วเชิงมุม
	# คำนวณ Jacobian สำหรับข้อต่อทั้ง 3 ข้อ
	for i in range(3):
		# ดึงค่าการหมุนและตำแหน่งของข้อต่อ i
		R_i  =  R[:, :, i]
		P_i  =  P[:, i]
		# หาแกนหมุนของข้อต่อ i
		z  =  R_i  @  np.array([0, 0, 1]) # แกนหมุนของข้อต่อนั้น
		# คำนวณความเร็วเชิงเส้นด้วยการทำ cross product ระหว่างแกนหมุนกับตำแหน่ง end effector
		J_v[:, i] =  np.cross(z, (p_e  -  P_i))
		# ความเร็วเชิงมุมก็คือแกนหมุน z ตรง ๆ
		J_w[:, i] =  z
	# รวมความเร็วเชิงเส้นกับความเร็วเชิงมุมในแนวตั้ง กลายเป็น Jacobian Matrix ขนาด 6x3
	J_e  =  np.vstack((J_v, J_w))
	return  J_e
```
### Example of use:
```bash
J_e = endEffectorJacobianHW3(q)
```

---
###  คำถามข้อที่ 2 : หาสภาวะ Singularity
### Determinant Condition

การตรวจการเข้าสู่สภาวะ (Singularity) ของหุ่นยน โดยเมื่อ determinant ของ Jacobian ส่วนความเร็วเชิงเส้น (Jv​) มีค่าเท่ากับศูนย์ จะบ่งชี้ว่าหุ่นยนต์อยู่ในสภาวะเอกพจน์ (singular configuration) ซึ่งหมายความว่าหุ่นยนต์สูญเสียความสามารถในการเคลื่อนที่ได้อย่างอิสระในบางทิศทาง

**Formula used in code:**
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/det1.png"  />
</p>

การที่ det⁡(Jv) = 0 บ่งบอกถึงสภาวะที่หุ่นยนต์เกิด (Singularity) ซึ่งหมายความว่าหุ่นยนต์ไม่สามารถเคลื่อนที่ได้อย่างอิสระในทุกทิศทาง 

### Method:
1.  **คำนวณเมทริกซ์ Jacobian**
นำค่า Jacobian matrix จากการหาในข้อแรก `J_e1 = endEffectorJacobianHW3(q)` แยกส่วนของความเร็วเชิงเส้น (Linear velocity Jacobian) เพื่อนำมาคำนวณค่า determinant โดยใช้ฟังก์ชัน `np.linalg.det`
2.  **การตรวจสอบค่า Determinant เพื่อหา Singularitie**
หากค่า determinant ของความเร็วเชิงเส้น น้อยกว่า `epsilon` ที่กำหนด (0.001) หุ่นยนต์จะถือว่าอยู่ในสถานะ Singular (flag = 1) มิฉะนั้นจะไม่อยู่ในสถานะ Singular (flag = 0) 
### ฟังก์ชั่นในการหา Singularity ของหุ่นยนต์ :
```bash
def checkSingularityHW3(q: list[float]) -> bool:
	epsilon  =  0.001  # กำหนดค่าวิกฤตเล็ก ๆ เพื่อใช้ตรวจสอบ singularity
	# คำนวณ Jacobian Matrix สำหรับ q ที่ส่งเข้ามา
	J_e1  =  endEffectorJacobianHW3(q)
	# ตัดเอาส่วนบน 3x3 ของ Jacobian ที่เกี่ยวกับความเร็วเชิงเส้น (J_v)
	J_v1  =  J_e1[:3, :3]
	# คำนวณค่า determinant ของ J_v1 เพื่อดูว่าเป็น 0 หรือไม่ (บ่งบอก singularity)
	det_J1  =  np.linalg.det(J_v1)
	# ถ้า determinant ใกล้ศูนย์มาก ถือว่าอยู่ในสถานะ singular
	if abs(det_J1) < epsilon:
		flag  =  1  # อยู่ในสภาวะ Singular
	else:
		flag  =  0  # ไม่อยู่ในสภาวะ Singular
	return flag # คืนค่าผลลัพธ์ว่าหุ่นอยู่ในสถานะ singular หรือไม่
```
### Example of use:
```bash
singularity = checkSingularityHW3(q)
```

---
###  คำถามข้อที่ 3 : หา effort ของแต่ละข้อต่อเมื่อมี wrench มากระทำ

### Jacobian Transpose Method
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
นำค่า Jacobian ที่คำนวนจากข้อแรก `J_e = endEffectorJacobianHW3(q)`มาทำการ Transpose ได้ Jacobian Transpose (`J_e.T`) และนำมาคูณกับแรงภายนอก `w` ที่กระทำต่อ end effector เพื่อหาค่า torque ที่ข้อต่อต้องออกแรง
### ฟังก์ชั่นในการหา Torque ของหุ่นยนต์ :
```bash
def computeEffortHW3(q: list[float], w: list[float]) -> list[float]:
	# เรียกใช้ฟังก์ชันเพื่อคำนวณ Jacobian Matrix ของหุ่นยนต์ที่ตำแหน่งข้อต่อ q
	J_e  =  endEffectorJacobianHW3(q)
	# คำนวณ torque หรือความพยายามที่ข้อต่อต้องออกแรง (แรงบิด) โดยใช้ Jacobian Transpose (J_e.T) คูณกับแรง w
	tau  =  J_e.T @ w  # คำนวณ effort (tau)
	# ส่งค่า torque ที่คำนวณได้กลับไป
	return tau
```
### Example of use:
```bash
tua = computeEffortHW3(q,w)
```

## เช็คคำตอบ



### คำถามข้อที่ 1 : หา Jacobian  Matrix ของหุ่นยนต์
ในการตรวจคำตอบข้อนี้เราจะทำการใช้ `roboticstoolbox` เพื่อเป็นอีกวิธีคิดในการหา Jacobian matrix และนำมาเปรียบเทียบค่ากัน Jacobian matrix ที่คิดก่อนหน้าโดยจะต้องมีค่าความแตกต่างกันไม่เกิน 0.000001 จึงจะถูกต้อง
### ขั้นตอนการทำ

1.  **การหาพารามิเตอร์ DH**
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/Dh1.jpg"/>
</p>

**DH Parameter:**
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/Dh5.png"/>
</p>

2. **การสร้างโมเดลหุ่นยนต์**	
โดยจะใช้ `roboticstoolbox` เพื่อสร้างโมเดลหุ่นยนต์ โดยมีการสร้าง `DHRobot` ที่ประกอบด้วย Joint แต่ละตัวที่ระบุด้วยพารามิเตอร์ DH
3. **การคำนวณ Jacobian**

ใช้ฟังก์ชัน `jacob0()` จาก Robotics Toolbox เพื่อคำนวณ Jacobian ที่ตำแหน่งข้อต่อที่กำหนด

4. **การตรวจสอบผลลัพธ์**	
เราจะใช้การเปรียบเทียบผลลัพธ์ Jacobian ที่คำนวณเอง `J_ours = endEffectorJacobianHW3(q)` และ Jacobian ที่คำนวณจาก Robotics Toolbox `J_rtb = robot.jacob0(q)` เพื่อเช็คความถูกต้องของการคำนวณ โดยจะต้องมีค่าความแตกต่างกันไม่เกิน `0.000001`จึงจะแสดงค่า `True`

### Result#1
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/Result1.png"/>
</p>
จากผลการทดสอบ Jacobian Matrix ค่าที่ได้จากการคำนวณด้วยวิธีของเราและค่าจาก Robotics Toolbox ใกล้เคียงกันมาก ซึ่งหมายความว่าการคำนวณ Jacobian ของเรานั้นถูกต้องและแม่นยำ

---

###  คำถามข้อที่ 2 : หาสภาวะ Singularity

### ขั้นตอนการทำ
1.  **การคำนวณ Jacobian Matrix**
 คำนวณหาค่า Jacobian matrix จากทั้งวิธีของเราและ Robotics Toolbox
2.  **การหาค่า Determinant**    
ดึงส่วนความเร็วเชิงเส้น (Jv​) และคำนวณค่า determinant ด้วย `np.linalg.det(J_v)`
3.  **การตรวจสอบ Singular**    
ใช้เงื่อนไขเพื่อตรวจสอบว่าค่า determinant มีค่าน้อยกว่า 0.0001 หรือไม่ ถ้าใช่แสดงว่า Jacobian ใกล้เคียงกับสถานะเอกพจน์		
4.  **การสุ่มค่า q**
สุ่มค่าของ q และใช้ทั้งฟังก์ชันที่เราสร้างขึ้นและ Robotics Toolbox เพื่อตรวจสอบ

### Result#2
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/Result2.png"/>
</p>
จากผลการทดสอบ ค่า Singularity เห็นได้ว่า flag ที่แสดงมีค่าเท่ากัน ทั้งจากส่วนที่คำนวณด้วย DH Parameter และ Robotics Toolbox จึงสรุปได้ว่าผลการคำนวณถูกต้อง

---

###  คำถามข้อที่ 3 : หา effort ของแต่ละข้อต่อเมื่อมี wrench มากระทำ

### ขั้นตอนการทำ
1.  **การคำนวณ Jacobian Matrix**
ใช้สองวิธี ทั้งวิธีการคำนวณเองและการใช้ Robotics Toolbox เพื่อหาค่า Jacobian ของหุ่นยนต์
2.  **คำนวณ torque**
ใช้ τ=JT⋅w เพื่อหาค่า torque จาก Jacobian ทั้งสองวิธี
3.  **การสุ่มค่า q และ w**
สุ่มค่า q (มุมข้อต่อ) และ w (แรง/โมเมนต์) เพื่อทดสอบการคำนวณ
4.  **เปรียบเทียบผลลัพธ์**
 ตรวจสอบว่า torque จากทั้งสองวิธีใกล้เคียงกันหรือไม่โดยมีค่าต่างกันไม่เกิน 10^-6 จึงจะถือว่าตรงกัน

### Result#3
<p  align="center">
<img  src="http://cdn.jsdelivr.net/gh/whyawayme/image/Result3.png"/>
</p>
จากผลการทดสอบ Effort หรือค่า torque ที่ได้จากการคำนวณด้วยวิธีของเราและค่าจาก Robotics Toolbox ใกล้เคียงกันมาก ทั้งในส่วนของแต่ละข้อต่อ จึงสรุปได้ว่าผลการคำนวณ Effort ของเราถูกต้องและแม่นยำ
