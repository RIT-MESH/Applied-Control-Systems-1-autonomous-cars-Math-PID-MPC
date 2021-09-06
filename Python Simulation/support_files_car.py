'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin
2. This Python code can be freely used and distributed
3. The copyright label in this Python file such as

copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)
that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author
of this Python file be held responsible for whatever happens in relation to this Python file.
For example, if there is a bug in the code and because of that a project, invention,
or anything else it was used for fails - the author is NOT RESPONSIBLE!

'''

import numpy as np
import matplotlib.pyplot as plt

class SupportFilesCar:
    ''' The following functions interact with the main file'''

    def __init__(self):
        ''' Load the constants that do not change'''

        # Constants
        m=1500
        Iz=3000
        Caf=19000
        Car=33000
        lf=2
        lr=3
        Ts=0.02

        # Parameters for the lane change: [psi_ref 0;0 Y_ref]
        # Higher psi reduces the overshoot
        # Matrix weights for the cost function (They must be diagonal)
        Q=np.matrix('1 0;0 1') # weights for outputs (all samples, except the last one)
        S=np.matrix('1 0;0 1') # weights for the final horizon period outputs
        R=np.matrix('1') # weights for inputs (only 1 input in our case)

        outputs=2 # number of outputs
        hz = 20 # horizon period
        x_dot=20 # car's longitudinal velocity
        lane_width=7 # [m]
        nr_lanes=5 # 6 lanes [m]
        # r=14*np.random.randint(2)-7 # amplitude for sinusoidal functions
        # f=0.01*np.random.randint(2)+0.01 # frequency
        r=4
        f=0.01
        time_length = 10 # [s] - duration of the entire manoeuvre

        ### PID ###
        PID_switch=0 # Turn PID function ON/OFF (ON=1, OFF=0)

        Kp_yaw=7
        Kd_yaw=3
        Ki_yaw=5

        Kp_Y=7
        Kd_Y=3
        Ki_Y=5

        ### PID END ###

        trajectory=3 # You should only choose 1,2,3

        self.constants=[m, Iz, Caf, Car, lf, lr, Ts, Q, S, R, outputs, hz, x_dot, r, f, time_length,lane_width,PID_switch,Kp_yaw,Kd_yaw,Ki_yaw,Kp_Y,Kd_Y,Ki_Y,trajectory]

        return None

    def trajectory_generator(self,t,r,f):
        '''This method creates the trajectory for a car to follow'''

        Ts=self.constants[6]
        x_dot=self.constants[12]
        trajectory=self.constants[24]

        # Define the x length, depends on the car's longitudinal velocity
        x=np.linspace(0,x_dot*t[-1],num=len(t))

        # Define trajectories

        if trajectory==1:
            y=-9*np.ones(len(t))
        elif trajectory==2:
            y=9*np.tanh(t-t[-1]/2)
        elif trajectory==3:
            aaa=-28/100**2
            aaa=aaa/1.1
            if aaa<0:
                bbb=14
            else:
                bbb=-14
            y_1=aaa*(x+self.constants[16]-100)**2+bbb
            y_2=2*r*np.sin(2*np.pi*f*x)
            y=(y_1+y_2)/2
        else:
            print("For trajectories, only choose 1, 2, or 3 as an integer value")
            exit()

        # Vector of x and y changes per sample time
        dx=x[1:len(x)]-x[0:len(x)-1]
        dy=y[1:len(y)]-y[0:len(y)-1]

        # Define the reference yaw angles
        psi=np.zeros(len(x))
        psiInt=psi
        psi[0]=np.arctan2(dy[0],dx[0])
        psi[1:len(psi)]=np.arctan2(dy[0:len(dy)],dx[0:len(dx)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]

        return psiInt,x,y

    def state_space(self):
        '''This function forms the state space matrices and transforms them in the discrete form'''

        # Get the necessary constants
        m=self.constants[0]
        Iz=self.constants[1]
        Caf=self.constants[2]
        Car=self.constants[3]
        lf=self.constants[4]
        lr=self.constants[5]
        Ts=self.constants[6]
        x_dot=self.constants[12]

        # Get the state space matrices for the control

        A1=-(2*Caf+2*Car)/(m*x_dot)
        A2=-x_dot-(2*Caf*lf-2*Car*lr)/(m*x_dot)
        A3=-(2*lf*Caf-2*lr*Car)/(Iz*x_dot)
        A4=-(2*lf**2*Caf+2*lr**2*Car)/(Iz*x_dot)

        A=np.array([[A1, 0, A2, 0],[0, 0, 1, 0],[A3, 0, A4, 0],[1, x_dot, 0, 0]])
        B=np.array([[2*Caf/m],[0],[2*lf*Caf/Iz],[0]])
        C=np.array([[0, 1, 0, 0],[0, 0, 0, 1]])
        D=0

        # Discretise the system (forward Euler)
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad, Bd, Cd, Dd

    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz):
        '''This function creates the compact matrices for Model Predictive Control'''
        # db - double bar
        # dbt - double bar transpose
        # dc - double circumflex

        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd


        Q=self.constants[7]
        S=self.constants[8]
        R=self.constants[9]

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)


        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc

    def open_loop_new_states(self,states,U1):
        '''This function computes the new state vector for one sample time later'''

        # Get the necessary constants
        m=self.constants[0]
        Iz=self.constants[1]
        Caf=self.constants[2]
        Car=self.constants[3]
        lf=self.constants[4]
        lr=self.constants[5]
        Ts=self.constants[6]
        x_dot=self.constants[12]

        current_states=states
        new_states=current_states
        y_dot=current_states[0]
        psi=current_states[1]
        psi_dot=current_states[2]
        Y=current_states[3]

        sub_loop=30  #Chop Ts into 30 pieces
        for i in range(0,sub_loop):
            # Compute the the derivatives of the states
            y_dot_dot=-(2*Caf+2*Car)/(m*x_dot)*y_dot+(-x_dot-(2*Caf*lf-2*Car*lr)/(m*x_dot))*psi_dot+2*Caf/m*U1
            psi_dot=psi_dot
            psi_dot_dot=-(2*lf*Caf-2*lr*Car)/(Iz*x_dot)*y_dot-(2*lf**2*Caf+2*lr**2*Car)/(Iz*x_dot)*psi_dot+2*lf*Caf/Iz*U1
            Y_dot=np.sin(psi)*x_dot+np.cos(psi)*y_dot

            # Update the state values with new state derivatives
            y_dot=y_dot+y_dot_dot*Ts/sub_loop
            psi=psi+psi_dot*Ts/sub_loop
            psi_dot=psi_dot+psi_dot_dot*Ts/sub_loop
            Y=Y+Y_dot*Ts/sub_loop

        # Take the last states
        new_states[0]=y_dot
        new_states[1]=psi
        new_states[2]=psi_dot
        new_states[3]=Y

        return new_states
