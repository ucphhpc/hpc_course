#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>
#include <map>
#include <iomanip>
#include <valarray>
#include <stdio.h>
#include <random>



/* atom object  */
class Atom {
public:
    // The mass of the atom in (U)
    double mass;
    // The position of the bead in the x, y, and z axis in (nm)
    double m_px;
    double m_py;
    double m_pz;
    // The velocity of the bead in the x, y, and z axis (nm/ps)
    double m_vx;
    double m_vy;
    double m_vz;
    // The forces on the bead in the x, y, and z axis   (k_BT/nm)
    double m_fx;
    double m_fy;
    double m_fz;
    
    
    double m_ep;            // epsilon for LJ potential
    double m_sigma;         // Sigma, somehow the size of the atom
    double charge;          // charge of the atom (partial charge)
    const char *m_name;     // Name of the atom
};
class Bond {
public:
    // an object of a bond between two atoms U = 0.5k(r12-L0)^2
    double K;       // force constant
    double L0;
    int a1;
    int a2;


};
class Angle {
public:
    // an object of an angle between three atoms  U=0.5K(phi123-phi0)^2
    double K;
    double Phi0;
    int a1;
    int a2;
    int a3;

    
};
class Molecule {
public:
    // molecule object
    std::vector<Atom> m_Atoms;          // all the atoms in the molecule
    std::vector<Bond> m_Bonds;          // all the bonds, to create bond potentials
    std::vector<Angle> m_Angles;        // all the angles

};
class System {
public:
    std::vector<Molecule> m_molecules; // all the molecules in the system
    
    // system box size. for this code these values only used for vmd, but in general md codes, PBC exist
    double m_Lx;
    double m_Ly;
    double m_Lz;
    
};
class Sim_Configuration {
public:
    // simulation configurations: first step, last setp, number of the molecules in the system, time step
    // and final file name
    Sim_Configuration(std::vector <std::string>);
    ~Sim_Configuration();
    
    int i_step;
    int f_step;
    int no_mol;
    double dt;
    int data_period;
    std::string gro_filename;
    double T;
    double m_Lx;
    double m_Ly;
    double m_Lz;    
};
// 3d vector object for vector calculations; only needed for angle force
class Vec3D
{
public:
    Vec3D(double x,double y,double z);
    Vec3D();
    ~Vec3D();
    
private:
    double m_X;
    double m_Y;
    double m_Z;
    // vector operators
public:
    double& operator()(const int n);
    Vec3D operator + (Vec3D);
    Vec3D operator - (Vec3D);
    Vec3D operator * (Vec3D); //cross product
    Vec3D operator * (double );
    void operator = (Vec3D);
    double dot(Vec3D,Vec3D);  // dot product
    double norm();             // size of the vector
};
void Evolve(System&, Sim_Configuration&);  // integrating the system for one step
void UpdateForces(System&);                 // updating the forces on each particle based on the particles positions
System GenWaterBox(int N, double Lx, double Ly, double Lz);     // creating a system with N water molecules
Molecule MakeOneWater(double, double, double );                 // generating one water molecules
void WriteGro(System, std::string filename, char state);         // writing the system configurations in the trajectory file.
//======================================================================================================
//======================== Main function ======================================================================
//======================================================================================================
int main(int argc, char* argv[])
{

    std::vector <std::string> argument;
    std::string str;
    for (long i=0;i<argc;i++)
    {
        str.assign(argv[i]);
        argument.push_back(str);
    }
    Sim_Configuration SC(argument); // Update the system configuration from command line data
    
    System SYS  = GenWaterBox(SC.no_mol, SC.m_Lx,SC.m_Ly,SC.m_Lz);   // this will create a system containing SC.no_mol water molecules in a box of 40nm^3
    WriteGro(SYS, SC.gro_filename, 'f');                // writing the initial configuration in the trajectory file
    
    // MD simulation
    for (int step = SC.i_step;step<SC.f_step+1 ; step++)
    {
         Evolve(SYS, SC); // evolving the system by one step
        if(step%(SC.data_period)==0)
        {
            //writing the configuration in the trajectory file
            WriteGro(SYS, SC.gro_filename, 'i');
        }
    }
}
//================================================================================
//==========>> implementation of the functions <<========================================
//================================================================================

Molecule MakeOneWater(double X0, double Y0 , double Z0)
{
    //===========================================================
    // creating one water molecule at position X0,Y0,Z0. 3 atoms
    //                        H---O---H
    // The angle is 104.45 degrees and bond length is 0.09584 nm
    //===========================================================
    Molecule Water;
    double pi = acos(-1);
    std::vector<Atom> wateratom;
    
    // Oxygen atom
    Atom Oatom;
    Oatom.m_px = X0;
    Oatom.m_py = Y0;
    Oatom.m_pz = Z0;
    Oatom.mass = 16; // dalton
    // we set the initial velocity to zero for all the atoms. we could also give the particles a velocity with boltzmann distribution
    Oatom.m_vx = 0;
    Oatom.m_vy = 0;
    Oatom.m_vz = 0;
    Oatom.m_fx = 0;
    Oatom.m_fy = 0;
    Oatom.m_fz = 0;
    Oatom.m_ep = 0.65;
    Oatom.m_sigma = 0.31;
    Oatom.charge = -0.82;
    Oatom.m_name = "O";
    wateratom.push_back(Oatom);
    double l0 = 0.09584;
    // hydrogen atom
    Atom H1atom;
    H1atom.m_px = X0+l0*sin(52.225/180.0*pi);
    H1atom.m_py = Y0+l0*cos(52.225/180.0*pi);
    H1atom.m_pz = Z0;
    H1atom.m_vx = 0;
    H1atom.m_vy = 0;
    H1atom.m_vz = 0;
    H1atom.m_fx = 0;
    H1atom.m_fy = 0;
    H1atom.m_fz = 0;
    H1atom.mass = 1; // dalton
    H1atom.m_ep = 0.18828;
    H1atom.m_sigma = 0.238;
    H1atom.charge =  0.41;
    H1atom.m_name = "H";
    wateratom.push_back(H1atom);
    // hydrogen atom
    Atom H2atom;
    H2atom.m_px = X0-l0*sin(52.225/180.0*pi);
    H2atom.m_py = Y0+l0*cos(52.225/180.0*pi);
    H2atom.m_pz = Z0;
    H2atom.m_vx = 0;
    H2atom.m_vy = 0;
    H2atom.m_vz = 0;
    H2atom.m_fx = 0;
    H2atom.m_fy = 0;
    H2atom.m_fz = 0;
    H2atom.mass = 1; // dalton
    H2atom.m_ep = 0.18828;
    H2atom.m_sigma = 0.238;
    H2atom.charge =  0.41;
    H2atom.m_name = "H";
    wateratom.push_back(H2atom);
    Water.m_Atoms = wateratom;
    
    std::vector<Bond> waterbonds;
    //bond beetween H1-O
    Bond B1;
    B1.K = 20000;
    B1.L0 = 0.09584;
    B1.a1 = 1;
    B1.a2 = 2;
    waterbonds.push_back(B1);
    //bond between H2-O
    Bond B2;
    B2.K = 20000;
    B2.L0 = 0.09584;
    B2.a1 = 1;
    B2.a2 = 3;
    waterbonds.push_back(B2);
    Water.m_Bonds = waterbonds;
    
    // angle between H O H
    std::vector<Angle> waterangle;
    Angle An;
    An.K = 1000;
    An.Phi0 = 104.45;
    An.a1 = 2;
    An.a2 = 1;
    An.a3 = 3;
    waterangle.push_back(An);
    Water.m_Angles = waterangle;
    
    return Water;
}
System GenWaterBox(int N, double Lx, double Ly, double Lz)
{
    //create a system containing N water mol
    System Sys;
    Sys.m_Lx = Lx;
    Sys.m_Ly = Ly;
    Sys.m_Lz = Lz;
    std::vector<Molecule> molecules;
    for (int i = 0; i<N; i++)
    {
        Molecule W = MakeOneWater(Lx/2+0.2*double(i), Ly/2+ 0.2*double(i), Lz/2);
        molecules.push_back(W);
    }
    Sys.m_molecules = molecules;
    return Sys;
}
void Evolve(System &sys, Sim_Configuration &sc)
{
    double dt = sc.dt;
    double Lx = sys.m_Lx;
    double Ly = sys.m_Ly;
    double Lz = sys.m_Lz;
    std::vector<Molecule > molecules = sys.m_molecules;

    for (std::vector<Molecule>::iterator it1 = (sys.m_molecules).begin() ; it1 != (sys.m_molecules).end(); ++it1)
    {
        for (std::vector<Atom>::iterator it = (it1->m_Atoms).begin() ; it != (it1->m_Atoms).end(); ++it)
        {
            double mass = it->mass;
            it->m_vx+= dt*(it->m_fx/mass); // Update the velocities
            it->m_vy+= dt*(it->m_fy/mass);
            it->m_vz+= dt*(it->m_fz/mass);
            it->m_px+= dt*it->m_vx;         // Update the positions
            it->m_py+= dt*it->m_vy;
            it->m_pz+= dt*it->m_vz;
            it->m_fx = 0;                   // set the forces zero, in next step we update them based on potentials
            it->m_fy = 0;
            it->m_fz = 0;
        }
    }

    // Update the forces using the potentials
                UpdateForces(sys);

}
void UpdateForces(System &sys)
{
    double pi = acos(-1);
    
    //====  bond forces  (H--O bonds), U_bond = 0.5*k_b(r-l0)^2
    for (std::vector<Molecule>::iterator it1 = (sys.m_molecules).begin() ; it1 != (sys.m_molecules).end(); ++it1)
    {
        std::vector<Bond> Bonds = it1->m_Bonds;
        for (std::vector<Bond>::iterator it = Bonds.begin() ; it != Bonds.end(); ++it)
        {
            int i = it->a1-1;
            int j = it->a2-1;
            double K = it->K;
            double L0 = it->L0;
            
            Atom &bead1 = (it1->m_Atoms)[i];
            Atom &bead2 = (it1->m_Atoms)[j];
            
            double dx = (bead1.m_px-bead2.m_px);
            double dy = (bead1.m_py-bead2.m_py);
            double dz = (bead1.m_pz-bead2.m_pz);
            double r2=dx*dx+dy*dy+dz*dz;
            double r=sqrt(r2);
            double fx = -K*(1-L0/r)*dx;
            double fy = -K*(1-L0/r)*dy;
            double fz = -K*(1-L0/r)*dz;
            
            bead1.m_fx+=fx;
            bead1.m_fy+=fy;
            bead1.m_fz+=fz;
            
            bead2.m_fx-=fx;
            bead2.m_fy-=fy;
            bead2.m_fz-=fz;
            
        }
        
    }
    
    //====  angle forces  (H--O---H bonds) U_angle = 0.5*k_a(phi-phi_0)^2
     //f_H1 = -K(phi-ph0)/|H1O|*Ta
    // f_H2 = -K(phi-ph0)/|H2O|*Tc
    // f_O = -f1 - f2
    // Ta = norm(H1O*(H1O*H2O))
    // Tc = norm(H2O*(H2O*H1O))
    //=============================================================
    for (std::vector<Molecule>::iterator it1 = (sys.m_molecules).begin() ; it1 != (sys.m_molecules).end(); ++it1)
    {
        std::vector<Angle> angle = it1->m_Angles;
        for (std::vector<Angle>::iterator it = angle.begin() ; it != angle.end(); ++it)
        {
            int i = it->a1-1;
            int j = it->a2-1;
            int k = it->a3-1;
            
            double K = it->K;
            double Phi0 = (it->Phi0)*pi/180;
            
            Atom &bead1 = (it1->m_Atoms)[i];
            Atom &bead2 = (it1->m_Atoms)[j];
            Atom &bead3 = (it1->m_Atoms)[k];
            
            
            double dx21 = (bead2.m_px-bead1.m_px);
            double dy21 = (bead2.m_py-bead1.m_py);
            double dz21 = (bead2.m_pz-bead1.m_pz);
            
            double norm1 = sqrt(dx21*dx21+dy21*dy21+dz21*dz21);
            Vec3D dr21(dx21,dy21,dz21);
            
            double dx23 = (bead2.m_px-bead3.m_px);
            double dy23 = (bead2.m_py-bead3.m_py);
            double dz23 = (bead2.m_pz-bead3.m_pz);
            
            double norm2 = sqrt(dx23*dx23+dy23*dy23+dz23*dz23);
            Vec3D dr23(dx23,dy23,dz23);
            
            
            double phi = (dx21*dx23+dy21*dy23+dz21*dz23)/(norm1*norm2);
            
            phi = acos(phi);
            Vec3D Ta = dr21*(dr21*dr23);
            
            Ta = Ta*(1/Ta.norm());
            
            Vec3D Tc = dr23*(dr23*dr21);
            Tc = Tc*(1/Tc.norm());
            
            double f1 = K*(phi-Phi0)/norm1;
            double f3 = K*(phi-Phi0)/norm2;
            
            Vec3D F1 = Ta*f1;
            Vec3D F3 = Tc*f3;
            Vec3D F2 = (F1+F3)*(-1);
            bead1.m_fx+=F1(0);
            bead1.m_fy+=F1(1);
            bead1.m_fz+=F1(2);
            
            bead2.m_fx+=F2(0);
            bead2.m_fy+=F2(1);
            bead2.m_fz+=F2(2);
            
            bead3.m_fx+=F3(0);
            bead3.m_fy+=F3(1);
            bead3.m_fz+=F3(2);

        }
    }
    
    // nonbonded forces: We do not include any nonbonded force between atoms in the same molecules.
    // The total non-bonded forces come from Lennard Jones (LJ) and coulomb interactions
    // U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r
    
    for (int i = 0; i<(sys.m_molecules).size();i++)
        for (int j = i+1; j<(sys.m_molecules).size();j++)
        {
            int s1 = (((sys.m_molecules)[i]).m_Atoms).size();
            int s2 = (((sys.m_molecules)[j]).m_Atoms).size();
            
            for (int n = 0; n<s1; n++)
                for (int m = 0; m<s2; m++)
                {
                    
                    Atom &bead1 = (((sys.m_molecules)[i]).m_Atoms)[n];
                    Atom &bead2 = (((sys.m_molecules)[j]).m_Atoms)[m];
                    
                    double dx = (bead1.m_px-bead2.m_px);
                    double dy = (bead1.m_py-bead2.m_py);
                    double dz = (bead1.m_pz-bead2.m_pz);
                    
                    double r2=dx*dx+dy*dy+dz*dz;
                    double r=sqrt(r2);
                    double  ep = sqrt(bead1.m_ep*bead2.m_ep); // ep = sqrt(ep1*ep2)
                    double  sigma = 0.5*(bead1.m_sigma+bead2.m_sigma);  // sigma = (sigma1+sigma2)/2
                    double q1 = bead1.charge;           // q1
                    double q2 = bead2.charge;           // q2
                    
                    double sir = sigma*sigma/r2;
                    double KC = 80*0.7;
                    double fx = ep*(12*pow(sir,6)-6*pow(sir,3))*dx*sir + KC*q1*q2/r2*dx/r;
                    double fy = ep*(12*pow(sir,6)-6*pow(sir,3))*dy*sir + KC*q1*q2/r2*dy/r;
                    double fz = ep*(12*pow(sir,6)-6*pow(sir,3))*dz*sir + KC*q1*q2/r2*dz/r;
                    bead1.m_fx+=fx;
                    bead1.m_fy+=fy;
                    bead1.m_fz+=fz;
                    bead2.m_fx-=fx;
                    bead2.m_fy-=fy;
                    bead2.m_fz-=fz;
                }
        }
}
Sim_Configuration::Sim_Configuration(std::vector <std::string> argument)
{
        i_step = 0;
        f_step = 10000;                    // number of the steps
        data_period = 100;                  //  how often to save coordinate to trajectory
        dt = 0.0005;                        //  integrator time step
        gro_filename = "trajectory.gro";    // name of the output trajectory
        T = 1;                              // temperature, not used in this code
        no_mol = 4;                         // number of the water molecules in the system
        m_Lx = 40;
        m_Ly = 40;
        m_Lz = 40;

        for (long i=1;i<argument.size();i=i+2)
        {
            std::string Arg1 = argument.at(i);
            if(argument[i]=="-h")
            {
                // Write help
                exit(0);
                break;
            }
            else if(argument[i]=="-f")
            {
                f_step = std::stoi(argument[i+1]);
            }
            else if(argument[i]=="-no_mol")
            {
                no_mol = std::stoi(argument[i+1]);
            }
            else if(argument[i]=="-fwrite")
            {
                data_period = std::stoi(argument[i+1]);
            }
            else if(argument[i]=="-dt")
            {
                dt = std::stof(argument[i+1]);
            }
            else if(argument[i]=="-temp")
            {
                T = std::stof(argument[i+1]);
            }
            else if(argument[i]=="-box")
            {
                m_Lx = std::stof(argument[i+1]);
                m_Ly = std::stof(argument[i+2]);
                m_Lz = std::stof(argument[i+3]);
                i++;
                i++;

            }
            else if(argument[i]=="-ofile")
            {
               gro_filename = argument[i+1];
            }
            else
            {
                std::cout<<"---> error: the argument type is not recognized "<<std::endl;
            }
        }
    dt = dt/1.57350; /// convert to ps based on having energy in k_BT, and length in nm
        
}
Sim_Configuration::~Sim_Configuration()
{
}
void WriteGro(System sys, std::string filename, char state )
{
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    if(ext!="gro")
        filename=filename+".gro";
    
    FILE *fgro;
    if(state=='f')
        fgro = fopen(filename.c_str(), "w");
    else
        fgro = fopen(filename.c_str(), "a");
    
    double Lx = sys.m_Lx;
    double Ly = sys.m_Ly;
    double Lz = sys.m_Lz;
    std::vector<Molecule> molecules = sys.m_molecules;
    int totalno = 0;
    for (std::vector<Molecule>::iterator it1 = molecules.begin() ; it1 != molecules.end(); ++it1)
    {
        std::vector<Atom> Atoms = it1->m_Atoms;
        for (std::vector<Atom>::iterator it = Atoms.begin() ; it != Atoms.end(); ++it)
        {
            totalno++;
        }
    }
    
    const char* Title="dmc gmx file handler";
    int Size = 9 ;
    fprintf(fgro,  "%s\n",Title);
    fprintf(fgro, "%5d\n",totalno);
    int i=0;
    
    
    for (std::vector<Molecule>::iterator it1 = molecules.begin() ; it1 != molecules.end(); ++it1)
    {
        std::vector<Atom> Atoms = it1->m_Atoms;
        for (std::vector<Atom>::iterator it = Atoms.begin() ; it != Atoms.end(); ++it)
        {
            double x=it->m_px;
            double y=it->m_py;
            double z=it->m_pz;
            const char* A1=it->m_name;
            const char* A2=it->m_name;
            fprintf(fgro, "%5d%5s%5s%5d%8.3f%8.3f%8.3f\n",1,A1,A2,i++,x,y,z );
        }
        
    }
    fprintf(fgro,  "%10.5f%10.5f%10.5f\n",Lx,Ly,Lz);
    fclose(fgro);
    
}
//==== end

//============== Vec3D
Vec3D::Vec3D(double x,double y,double z)
{
    m_X=x;    m_Y=y;    m_Z=z;
}
Vec3D::Vec3D()
{
    m_X=0.0; m_Y=0.0; m_Z=0.0;
}
Vec3D::~Vec3D()
{
}
double& Vec3D::operator()(const int n)
{
    double *Value=0;
    if(n==0)
    {
        Value=&m_X;
    }
    else if(n==1)
    {
        Value=&m_Y;
    }
    else if(n==2)
    {
        Value=&m_Z;
    }
    else
    {
        std::cout<<"Error: index should not be larger the 2 \n";
    }
    return *Value;
}
//-------------------------------------------------------
Vec3D Vec3D::operator+(Vec3D M)
{
    Vec3D M1;
    M1(0)=(*this)(0)+M(0);
    M1(1)=(*this)(1)+M(1);
    M1(2)=(*this)(2)+M(2);
    return M1;
}

//-------------------------------------------------------
Vec3D Vec3D::operator-(Vec3D M)
{
    Vec3D M1;
    M1(0)=(*this)(0)-M(0);
    M1(1)=(*this)(1)-M(1);
    M1(2)=(*this)(2)-M(2);
    return M1;
}
//-------------------------------------------------------
Vec3D Vec3D::operator*(Vec3D M)
{
    Vec3D M1;
    M1(0)=((*this)(1))*M(2)-((*this)(2))*M(1);
    M1(1)=((*this)(2))*M(0)-((*this)(0))*M(2);
    M1(2)=((*this)(0))*M(1)-((*this)(1))*M(0);
    
    return M1;
}
Vec3D Vec3D::operator*(double x)
{
    Vec3D M1;
    M1(0)=((*this)(0))*x;
    M1(1)=((*this)(1))*x;
    M1(2)=((*this)(2))*x;
    
    return M1;
}
void Vec3D::operator=(Vec3D M)
{
    m_X= M(0);
    m_Y= M(1);
    m_Z= M(2);
}
double Vec3D::norm ()
{
    double No=m_X*m_X+m_Y*m_Y+m_Z*m_Z;
    return sqrt(No);
}
double Vec3D::dot(Vec3D v1,Vec3D v2)
{
    double No=v1(0)*v2(0)+v1(1)*v2(1)+v1(2)*v2(2);
    return No;
}
