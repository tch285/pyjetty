#ifndef __PYJETTY_ECORREL_HH
#define __PYJETTY_ECORREL_HH

#include <vector>
#include <fastjet/PseudoJet.hh>
namespace EnergyCorrelators
{
	class CorrelatorsContainer
	{
		public:
			CorrelatorsContainer();
			CorrelatorsContainer(const int &ipt, const int &nparts, const int &ngroups);
			virtual ~CorrelatorsContainer();
			void addwr(const int &igroup, const double &w, const double &r);
			void addwr(const int &igroup, const double &w, const double &r, const int &indx1, const int &indx2);
			void clear();
			void PrintLists();
			std::vector<double> *weights();
			std::vector<double> *rs(); // return the pointer to the list of pair distances
			std::vector<double> *rxw();
			// have to use double instead of int or there will be errors later when accessing the indices
			std::vector<int> *indices1(); // indices of object 1 in the pair
			std::vector<int> *indices2(); // indices of object 2 in the pair
			std::vector<std::vector<int>> *indices();

			const double *wa();
			const double *ra();

		private:
			std::vector<double> fr; // list of pair distances
			std::vector<double> fw; // list of pair weights
			std::vector<double> frxw; // list of pair weights * distances
			std::vector<int> findx1; // list of indices of first contributing parton
			std::vector<int> findx2; // list of indices of second contributing parton
			std::vector<std::vector<int>> fidx; // list of all parton index combinations
			std::vector<std::vector<int>> fq; // list of charges of partons in each combination, currently unimplemented
			std::vector<int> fqprod; // list of charge products for each combination, curnrelty unimplemented
	};

	class CorrelatorBuilder
	{
		public:
			CorrelatorBuilder();
			// note by default we use energy correlators - one could use different weighting... future: pass as a param
			CorrelatorBuilder(const std::vector<fastjet::PseudoJet> &parts, const double &scale, const int &nmax, const int &power, const double dphi_cut, const double deta_cut);
			CorrelatorsContainer *correlator(int n);
			virtual ~CorrelatorBuilder();

			bool ApplyDeltaPhiRejection(const double dphi_cut, const double q1, const double q2, const double pt1, const double pt2, const double phi12);
			bool ApplyDeltaEtaRejection(const double deta_cut, const double eta12);
			int calcGroups(int, int);

		private:
			int fncmax;
			std::vector<CorrelatorsContainer*> fec;
	};

	std::vector<fastjet::PseudoJet> constituents_as_vector(const fastjet::PseudoJet &jet);

	std::vector<fastjet::PseudoJet> merge_signal_background_pjvectors(const std::vector<fastjet::PseudoJet> &signal, 
																	  const std::vector<fastjet::PseudoJet> &background,
																	  const double pTcut,
																	  const int bg_index_start);
};

#endif
