"""
Quadcopter PD Simulation — 3D Orientation + Live Telemetry + True Speed Control (Final)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider


# =========================================================
# PD Controller
# =========================================================
class QuadcopterPD:
    def __init__(self, m=0.028, l=0.046, b=1.4e-6, d=1.1e-7,
                 Ixx=16.6e-6, Iyy=16.6e-6, Izz=29.3e-6,
                 g=9.81, dt=0.002):
        self.m, self.l, self.b, self.d = m, l, b, d
        self.Ix, self.Iy, self.Iz = Ixx, Iyy, Izz
        self.g, self.dt = g, dt
        self.x = np.zeros(3); self.v = np.zeros(3)
        self.ang = np.zeros(3); self.rate = np.zeros(3)

        # PD gains
        self.Kp_z, self.Kd_z = 3.0, 1.2
        self.Kp_phi, self.Kd_phi = 1.5, 0.3
        self.Kp_theta, self.Kd_theta = 1.5, 0.3
        self.Kp_psi, self.Kd_psi = 1.2, 0.25
        self.Kp_xy = np.array([0.8, 0.8])
        self.Kd_xy = np.array([0.6, 0.6])

        # Limits
        self.U1_min = 0.3*m*g; self.U1_max = 1.7*m*g
        self.tau_max = 3e-4
        self.M = np.array([[ b,  b,  b,  b ],
                           [ 0, -l*b, 0,  l*b ],
                           [ -l*b, 0,  l*b, 0 ],
                           [ d, -d,  d, -d ]])
        self.M_inv = np.linalg.inv(self.M)
        self.z_ref, self.psi_ref = 1.0, 0.0

    # --- Rotations ---
    @staticmethod
    def Rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    @staticmethod
    def Ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    @staticmethod
    def Rx(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def R_wb(self, phi, theta, psi):
        return self.Rz(psi) @ self.Ry(theta) @ self.Rx(phi)

    def reset(self):
        self.x[:] = self.v[:] = self.ang[:] = self.rate[:] = 0

    # --- Control + dynamics ---
    def compute_control(self, pos_ref, vel_ref):
        exy = pos_ref[:2]-self.x[:2]; evxy = vel_ref[:2]-self.v[:2]
        a_des_xy = self.Kp_xy*exy + self.Kd_xy*evxy
        ez = self.z_ref-self.x[2]; evz=-self.v[2]
        U1 = self.m*(self.g + self.Kp_z*ez + self.Kd_z*evz)
        U1 = np.clip(U1,self.U1_min,self.U1_max)

        psi=self.ang[2]
        phi_des=(a_des_xy[0]*np.sin(psi)-a_des_xy[1]*np.cos(psi))/self.g
        theta_des=(a_des_xy[0]*np.cos(psi)+a_des_xy[1]*np.sin(psi))/self.g

        phi,theta,psi=self.ang; p,q,r=self.rate
        tau_phi=self.Kp_phi*(phi_des-phi)+self.Kd_phi*(0-p)
        tau_theta=self.Kp_theta*(theta_des-theta)+self.Kd_theta*(0-q)
        tau_psi=self.Kp_psi*(self.psi_ref-psi)+self.Kd_psi*(0-r)
        tau_phi=np.clip(tau_phi,-self.tau_max,self.tau_max)
        tau_theta=np.clip(tau_theta,-self.tau_max,self.tau_max)
        tau_psi=np.clip(tau_psi,-self.tau_max,self.tau_max)
        return U1,tau_phi,tau_theta,tau_psi

    def motor_mix(self,U1,tphi,ttheta,tpsi):
        u=np.array([U1,tphi,ttheta,tpsi])
        w2=self.M_inv@u; w2=np.clip(w2,0.0,2500.0**2)
        return np.sqrt(w2)

    def dynamics(self,state,omega):
        x,y,z,vx,vy,vz,phi,theta,psi,p,q,r=state
        m,g=self.m,self.g; Ix,Iy,Iz=self.Ix,self.Iy,self.Iz
        b,d,l=self.b,self.d,self.l
        w1,w2,w3,w4=omega
        U1=b*(w1**2+w2**2+w3**2+w4**2)
        tphi=l*b*(w4**2-w2**2)
        ttheta=l*b*(w3**2-w1**2)
        tpsi=d*(w1**2-w2**2+w3**2-w4**2)
        R=self.R_wb(phi,theta,psi); zb=R@np.array([0,0,1])
        ax,ay,az=(U1*zb/m)[0],(U1*zb/m)[1],(U1*zb/m)[2]-g
        p_dot=((Iy-Iz)/Ix)*q*r+tphi/Ix
        q_dot=((Iz-Ix)/Iy)*p*r+ttheta/Iy
        r_dot=((Ix-Iy)/Iz)*p*q+tpsi/Iz
        cth=np.cos(theta)
        sphi,cphi=np.sin(phi),np.cos(phi)
        sth=np.sin(theta)
        cth=np.clip(cth,1e-3,None)
        T=np.array([[1,sphi*sth/cth,cphi*sth/cth],
                    [0,cphi,-sphi],
                    [0,sphi/cth,cphi/cth]])
        ang_dot=T@np.array([p,q,r])
        return np.array([vx,vy,vz,ax,ay,az,*ang_dot,p_dot,q_dot,r_dot])

    def step(self,pos_ref,vel_ref):
        U1,tphi,ttheta,tpsi=self.compute_control(pos_ref,vel_ref)
        omega=self.motor_mix(U1,tphi,ttheta,tpsi)
        state=np.hstack([self.x,self.v,self.ang,self.rate])
        f=lambda s:self.dynamics(s,omega); dt=self.dt
        k1=f(state); k2=f(state+0.5*dt*k1)
        k3=f(state+0.5*dt*k2); k4=f(state+dt*k3)
        state+=dt/6*(k1+2*k2+2*k3+k4)
        self.x,self.v,self.ang,self.rate=state[:3],state[3:6],state[6:9],state[9:12]


# =========================================================
def square_traj(t,side=1.0,period=16.0,z=1.0):
    T=period/4; tm=t%period
    if tm<T: s=tm/T; x,y,vx,vy=side*s,0,side/T,0
    elif tm<2*T: s=(tm-T)/T; x,y,vx,vy=side,side*s,0,side/T
    elif tm<3*T: s=(tm-2*T)/T; x,y,vx,vy=side*(1-s),side,-side/T,0
    else: s=(tm-3*T)/T; x,y,vx,vy=0,side*(1-s),0,-side/T
    return np.array([x,y,z]),np.array([vx,vy,0.0])


# =========================================================
def demo_square_fast():
    quad=QuadcopterPD(); quad.reset()
    Tn=int(20/quad.dt); t_arr=np.arange(Tn)*quad.dt
    pos,ang,ref=[],[],[]
    for t in t_arr:
        pr,vr=square_traj(t,1.0,16.0,1.0)
        quad.step(pr,vr); pos.append(quad.x.copy())
        ang.append(quad.ang.copy()); ref.append(pr.copy())
    pos,ang,ref=map(np.array,[pos,ang,ref])

    # --- Figure setup ---
    fig=plt.figure(figsize=(10,8))
    gs=fig.add_gridspec(3,1,height_ratios=[3,1,0.25])
    ax=fig.add_subplot(gs[0],projection='3d')
    ax_err=fig.add_subplot(gs[1]); ax_slider=fig.add_subplot(gs[2])

    ax.plot(ref[:,0],ref[:,1],ref[:,2],'r--',label='reference')
    line,=ax.plot([],[],[],'b-',lw=2,label='path')
    body_x,=ax.plot([],[],[],'k-',lw=3)
    body_y,=ax.plot([],[],[],'gray',lw=3)
    point,=ax.plot([],[],[],'bo',ms=6)
    txt=ax.text2D(0.05,0.95,"",transform=ax.transAxes)
    ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5); ax.set_zlim(0,1.6)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("Quadcopter 3D Path — Fast Mode")

    bars=ax_err.bar(["X","Y","Z"],[0,0,0],
                    color=["#1A66A4","#4CAF50","#FF5722"])
    ax_err.set_ylim(-0.5,0.5); ax_err.set_ylabel("Error [m]")
    slider=Slider(ax_slider,"Speed",0.5,5.0,valinit=1.0,valstep=0.1)
    speed_factor=[1.0]
    slider.on_changed(lambda v:speed_factor.__setitem__(0,v))

    # --- Update ---
    def update(frame):
        idx=int(frame*speed_factor[0]*5)%len(pos)
        x,y,z=pos[idx]; phi,th,psi=ang[idx]
        R=quad.R_wb(phi,th,psi)

        # trail
        start=max(0,idx-300)
        line.set_data(pos[start:idx+1,0],pos[start:idx+1,1])
        line.set_3d_properties(pos[start:idx+1,2])
        point.set_data([x],[y]); point.set_3d_properties([z])

        arm=0.05
        b_axes=np.array([[-arm,arm,0,0],[0,0,-arm,arm],[0,0,0,0]])
        b_axes=R@b_axes
        body_x.set_data([x+b_axes[0,0],x+b_axes[0,1]],
                        [y+b_axes[1,0],y+b_axes[1,1]])
        body_x.set_3d_properties([z+b_axes[2,0],z+b_axes[2,1]])
        body_y.set_data([x+b_axes[0,2],x+b_axes[0,3]],
                        [y+b_axes[1,2],y+b_axes[1,3]])
        body_y.set_3d_properties([z+b_axes[2,2],z+b_axes[2,3]])

        txt.set_text(f"X={x:.2f}  Y={y:.2f}  Z={z:.2f}")
        err=ref[idx]-pos[idx]
        for b,e in zip(bars,err): b.set_height(e)
        return line,body_x,body_y,point,txt,*bars

    # --- Animation (kept alive to avoid warning) ---
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(pos)//2),
        interval=1, blit=False
    )

    plt.tight_layout()
    plt.show()
    del ani  # clean teardown


if __name__=="__main__":
    demo_square_fast()
