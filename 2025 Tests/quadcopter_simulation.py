"""
Quadcopter PD Simulator — Live 3D Animation + Error Bars + PID Plot Dashboard
Author: Lawrence Wontumi (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider


# =========================================================
# 1. PD-Controlled Quadcopter Model
# =========================================================
class QuadcopterPD:
    def __init__(self, dt=0.002, g=9.81):
        self.m, self.l, self.b, self.d = 0.028, 0.046, 1.4e-6, 1.1e-7
        self.Ix, self.Iy, self.Iz = 16.6e-6, 16.6e-6, 29.3e-6
        self.g, self.dt = g, dt
        self.reset()

        # PD gains
        self.Kp_z, self.Kd_z = 3.0, 1.2
        self.Kp_phi, self.Kd_phi = 1.5, 0.3
        self.Kp_theta, self.Kd_theta = 1.5, 0.3
        self.Kp_psi, self.Kd_psi = 1.2, 0.25
        self.Kp_xy = np.array([0.8, 0.8])
        self.Kd_xy = np.array([0.6, 0.6])

        # limits
        self.U1_min = 0.3*self.m*g; self.U1_max = 1.7*self.m*g
        self.tau_max = 3e-4

        self.M = np.array([[ self.b,  self.b,  self.b,  self.b ],
                           [ 0, -self.l*self.b, 0,  self.l*self.b ],
                           [ -self.l*self.b, 0,  self.l*self.b, 0 ],
                           [ self.d, -self.d,  self.d, -self.d ]])
        self.M_inv = np.linalg.inv(self.M)

    def reset(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.ang = np.zeros(3)
        self.rate = np.zeros(3)

    @staticmethod
    def Rz(psi): c,s=np.cos(psi),np.sin(psi); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    @staticmethod
    def Ry(theta): c,s=np.cos(theta),np.sin(theta); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    @staticmethod
    def Rx(phi): c,s=np.cos(phi),np.sin(phi); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def R_wb(self,phi,theta,psi): return self.Rz(psi)@self.Ry(theta)@self.Rx(phi)

    def compute_control(self, pos_ref, vel_ref, z_ref=1.0, psi_ref=0.0):
        exy = pos_ref[:2]-self.x[:2]; evxy = vel_ref[:2]-self.v[:2]
        a_des_xy = self.Kp_xy*exy + self.Kd_xy*evxy

        ez = z_ref-self.x[2]; evz=-self.v[2]
        U1 = self.m*(self.g + self.Kp_z*ez + self.Kd_z*evz)
        U1=np.clip(U1,self.U1_min,self.U1_max)

        psi=self.ang[2]
        phi_des=(a_des_xy[0]*np.sin(psi)-a_des_xy[1]*np.cos(psi))/self.g
        theta_des=(a_des_xy[0]*np.cos(psi)+a_des_xy[1]*np.sin(psi))/self.g

        phi,theta,psi=self.ang; p,q,r=self.rate
        tau_phi=self.Kp_phi*(phi_des-phi)+self.Kd_phi*(0-p)
        tau_theta=self.Kp_theta*(theta_des-theta)+self.Kd_theta*(0-q)
        tau_psi=self.Kp_psi*(psi_ref-psi)+self.Kd_psi*(0-r)
        tau_phi=np.clip(tau_phi,-self.tau_max,self.tau_max)
        tau_theta=np.clip(tau_theta,-self.tau_max,self.tau_max)
        tau_psi=np.clip(tau_psi,-self.tau_max,self.tau_max)

        pid_terms = dict(
            px=self.Kp_xy[0]*exy[0], dx=self.Kd_xy[0]*evxy[0],
            py=self.Kp_xy[1]*exy[1], dy=self.Kd_xy[1]*evxy[1],
            pz=self.Kp_z*ez, dz=self.Kd_z*evz,
            ex=exy[0], ey=exy[1], ez=ez
        )
        return U1,tau_phi,tau_theta,tau_psi,pid_terms

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
        cth=np.cos(theta); sphi,cphi=np.sin(phi),np.cos(phi)
        sth=np.sin(theta); cth=np.clip(cth,1e-3,None)
        T=np.array([[1,sphi*sth/cth,cphi*sth/cth],
                    [0,cphi,-sphi],
                    [0,sphi/cth,cphi/cth]])
        ang_dot=T@np.array([p,q,r])
        return np.array([vx,vy,vz,ax,ay,az,*ang_dot,p_dot,q_dot,r_dot])

    def step(self,pos_ref,vel_ref):
        U1,tphi,ttheta,tpsi,pid=self.compute_control(pos_ref,vel_ref)
        omega=self.M_inv@np.array([U1,tphi,ttheta,tpsi])
        omega=np.sqrt(np.clip(omega,0,2500.0**2))
        state=np.hstack([self.x,self.v,self.ang,self.rate])
        f=lambda s:self.dynamics(s,omega); dt=self.dt
        k1=f(state); k2=f(state+0.5*dt*k1)
        k3=f(state+0.5*dt*k2); k4=f(state+dt*k3)
        state+=dt/6*(k1+2*k2+2*k3+k4)
        self.x,self.v,self.ang,self.rate=state[:3],state[3:6],state[6:9],state[9:12]
        return pid


# =========================================================
# 2. Simulator with live PID + Error Bars
# =========================================================
class QuadcopterSim:
    def __init__(self,trajectory_fn,dt=0.002):
        self.drone=QuadcopterPD(dt)
        self.trajectory_fn=trajectory_fn
        self.dt=dt
        self.results={}

    def simulate(self,t_final=20):
        quad=self.drone
        N=int(t_final/quad.dt)
        t=np.arange(N)*quad.dt
        pos=[];ang=[];ref=[];pid=[]
        for ti in t:
            pr,vr=self.trajectory_fn(ti)
            pid_terms=quad.step(pr,vr)
            pos.append(quad.x.copy()); ang.append(quad.ang.copy())
            ref.append(pr.copy()); pid.append(pid_terms)
        self.results=dict(t=t,pos=np.array(pos),ang=np.array(ang),
                          ref=np.array(ref),pid=pid)

    def animate(self,speed=5.0):
        pos,ang,ref=self.results["pos"],self.results["ang"],self.results["ref"]
        pid=self.results["pid"]; t=self.results["t"]

        fig=plt.figure(figsize=(10,9))
        gs=fig.add_gridspec(4,1,height_ratios=[3,0.8,0.8,0.3])
        ax3d=fig.add_subplot(gs[0],projection='3d')
        ax_err=fig.add_subplot(gs[1])
        ax_pid=fig.add_subplot(gs[2])
        ax_slider=fig.add_subplot(gs[3])

        # --- 3D setup ---
        ax3d.plot(ref[:,0],ref[:,1],ref[:,2],'r--')
        line,=ax3d.plot([],[],[],'b-',lw=2)
        body_x,=ax3d.plot([],[],[],'k-',lw=3)
        body_y,=ax3d.plot([],[],[],'gray',lw=3)
        point,=ax3d.plot([],[],[],'bo',ms=6)
        txt=ax3d.text2D(0.05,0.9,"",transform=ax3d.transAxes)
        ax3d.set_xlim(-1.2,1.2); ax3d.set_ylim(-1.2,1.2); ax3d.set_zlim(0,1.6)
        ax3d.set_title("Live 3D Flight + PID Dashboard")

        # --- error bars ---
        bars=ax_err.bar(["X","Y","Z"],[0,0,0],
                        color=["#1A66A4","#4CAF50","#FF5722"])
        ax_err.set_ylim(-0.5,0.5); ax_err.set_ylabel("Error [m]")

        # --- PID live plot setup ---
        ax_pid.set_xlim(0,t[-1]); ax_pid.set_ylim(-2,2)
        line_px,=ax_pid.plot([],[],'r-',label='P_x')
        line_dx,=ax_pid.plot([],[],'r--',label='D_x')
        line_py,=ax_pid.plot([],[],'g-',label='P_y')
        line_dy,=ax_pid.plot([],[],'g--',label='D_y')
        line_pz,=ax_pid.plot([],[],'b-',label='P_z')
        line_dz,=ax_pid.plot([],[],'b--',label='D_z')
        ax_pid.legend(loc='upper right')
        ax_pid.set_ylabel("PID terms"); ax_pid.grid(True)

        # --- Speed slider ---
        slider=Slider(ax_slider,"Speed",0.5,5.0,valinit=1.0,valstep=0.1)
        speed_factor=[1.0]
        slider.on_changed(lambda v:speed_factor.__setitem__(0,v))

        # --- update function ---
        def update(frame):
            idx=int(frame*speed_factor[0])%len(pos)
            x,y,z=pos[idx]; phi,th,psi=ang[idx]
            R=self.drone.R_wb(phi,th,psi)

            # trail
            line.set_data(pos[:idx+1,0],pos[:idx+1,1])
            line.set_3d_properties(pos[:idx+1,2])
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
            txt.set_text(f"X={x:.2f} Y={y:.2f} Z={z:.2f}")

            # error bars
            err=ref[idx]-pos[idx]
            for b,e in zip(bars,err): b.set_height(e)

            # PID live scroll
            time=t[:idx]; pid_data=pid[:idx]
            px=[p["px"] for p in pid_data]; dx=[p["dx"] for p in pid_data]
            py=[p["py"] for p in pid_data]; dy=[p["dy"] for p in pid_data]
            pz=[p["pz"] for p in pid_data]; dz=[p["dz"] for p in pid_data]
            line_px.set_data(time,px); line_dx.set_data(time,dx)
            line_py.set_data(time,py); line_dy.set_data(time,dy)
            line_pz.set_data(time,pz); line_dz.set_data(time,dz)
            return line,body_x,body_y,point,txt,*bars, \
                   line_px,line_dx,line_py,line_dy,line_pz,line_dz

        ani=animation.FuncAnimation(fig,update,
                frames=len(pos)//2,interval=1,blit=False)
        plt.tight_layout(); plt.show(); del ani
