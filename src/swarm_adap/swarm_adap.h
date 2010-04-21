/*
    Header file for the program swarm_adap.
    "swarm_adap" is a program that uses the Swarm-NG tools for modeling an ensemble of
    small N systems using the hermite_adap_gpu integrator.
    Copyright (C) 2010  Swarm-NG Development Group

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact aaron.boley at the domain gmail.com if you have questions regarding
    this software.
*/

#ifndef __SWARM_ADAP_H__
#define __SWARM_ADAP_H__

#ifndef OUTPUT_DIRECTORY 
#define OUTPUT_DIRECTORY "adap_output"
#endif
#define real double

#define GCGS     6.67e-8
#define AUCGS    1.496e13
#define yrToCodeTime (2.*M_PI)
#define secondsToCodeTime (2.*M_PI)/(365.25*24.*3600.)
#define kmpsToCodeVel (1./(AUCGS*1e-5*secondsToCodeTime))

// compute the energies of each system
double calc_system_energy(const cpu_ensemble &ens, const int sys)
{
  double E = 0.;
      for(int bod1 = 0; bod1 != ens.nbod(); bod1++)
	{
	  float m1; double x1[3], v1[3];
	  ens.get_body(sys, bod1, m1, x1[0], x1[1], x1[2], v1[0], v1[1], v1[2]);
	  E += 0.5*m1*(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
	  
	  for(int bod2 = 0; bod2 < bod1; bod2++)
	    {
	      float m2; double x2[3], v2[3];
	      ens.get_body(sys, bod2, m2, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2]);
	      double dist = sqrt((x2[0]-x1[0])*(x2[0]-x1[0])+(x2[1]-x1[1])*(x2[1]-x1[1])+(x2[2]-x1[2])*(x2[2]-x1[2]));
	      
	      E -= m1*m2/dist;
	    }
	}
	return E;
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Read observing file
vector<real> getObsTimes()
 {
     ifstream ThisFile;
     stringstream buffer;
     string line;
     vector<real> ObsTimes;

     buffer.str("");//clear buffer
     buffer<<"../ic"<<"/"<<"observeTimes.dat";
     ThisFile.open(buffer.str().c_str());
     assert(ThisFile.good());

     while(!ThisFile.eof()) 
      {
        stringstream value;
        float floatTemp;
        getline(ThisFile,line);
        value<<line;
        if (ThisFile.eof())continue;
        value>>floatTemp;
        ObsTimes.push_back(floatTemp*yrToCodeTime);
      }

#if VERBOSE_OUTPUT>0
     for(unsigned int i=0;i<ObsTimes.size();++i)
      {
        cout<<ObsTimes[i]<<" Observation Time (code)\n";
      }
#endif
     ThisFile.close();
     return ObsTimes;

 }
////////////////////////////////////////////////////////////////////////
#endif