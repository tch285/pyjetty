#include "ecorrel.hh"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fastjet/Selector.hh>

namespace EnergyCorrelators
{
    CorrelatorsContainer::CorrelatorsContainer()
    : fr()
    , fw()
    , frxw()
    , findx1()
    , findx2()
    {
        ;
    }

    CorrelatorsContainer::CorrelatorsContainer(const int &ipoint, const int &nparts, const int &ngroups)
    : fr(ngroups)
    , fw(ngroups)
    , frxw(ngroups)
    , findx1(ngroups)
    , findx2(ngroups)
    , fidx(ngroups, std::vector<int> (ipoint))
    , fq(ngroups, std::vector<int> (ipoint))
    , fqprod(ngroups)
    {
        // permutations WITH replacement
        int max = std::pow(nparts, ipoint);
        for (int igroup = 1; igroup < ngroups; igroup++) {
            int remainder{igroup};
            for (int i = ipoint - 1; i >= 0; i--) {
                // cout << remainder % nparts << endl;
                fidx[igroup][i] = remainder % nparts;
                remainder /= nparts;
            }
        }

        // combinations WITHOUT replacement:
        // if (ngroups > 0) {
        //     std::string bitmask(ipoint, 1); // ipoint leading 1's
        //     bitmask.resize(nparts, 0); // nparts - ipoint trailing 0's
        //     int group_idx{0}, part_idx{0}, bit_idx{0};

        //     do {
        //         while (part_idx < ipoint) {
        //             if (bitmask[bit_idx]) {
        //                 fidx[group_idx][part_idx] = bit_idx;
        //                 part_idx++;
        //             }
        //             bit_idx++;
        //         }
        //         group_idx++;
        //         part_idx = bit_idx = 0;
        //     } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
        // }

        // combinations WITH replacement:
        // std::string bitmask(ipoint, 1); // ipoint leading 1's
        // bitmask.resize(nparts + ipoint - 1, 0); // nparts - ipoint trailing 0's
        // int group_idx{0}, part_idx{0}, bit_idx{0}, focus_idx{0};
        // do {
        //     while (part_idx < ipoint) {
        //         if (bitmask[bit_idx]) {
        //             fidx[group_idx][part_idx] = focus_idx;
        //             part_idx++;
        //         }
        //         else focus_idx++;
        //         bit_idx++;
        //     }
        //     group_idx++;
        //     part_idx = bit_idx = focus_idx = 0;
        // } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    }
    
    CorrelatorsContainer::~CorrelatorsContainer()
    {
        ;
    }

    void CorrelatorsContainer::clear()
    {
        fw.clear();
        fr.clear();
        frxw.clear();
        findx1.clear();
        findx2.clear();
    }

    void CorrelatorsContainer::addwr(const int &igroup, const double &w, const double &r)
    {
        fw[igroup] = w;
        fr[igroup] = r;
    }

    void CorrelatorsContainer::addwr(const int &igroup, const double &w, const double &r, const int &indx1, const int &indx2)
    {
        fw[igroup] = w;
        fr[igroup] = r;
        findx1[igroup] = indx1;
        findx2[igroup] = indx2;
    }

    std::vector<double> *CorrelatorsContainer::weights()
    {
        return &fw;
    }

    std::vector<double> *CorrelatorsContainer::rs()
    {
        return &fr;
    }

    std::vector<int> *CorrelatorsContainer::indices1()
    {
        return &findx1;
    }

    std::vector<int> *CorrelatorsContainer::indices2()
    {
        return &findx2;
    }

    std::vector<std::vector<int>> *CorrelatorsContainer::indices()
    {
        return &fidx;
    }

    const double *CorrelatorsContainer::wa()
    {
        return &fw[0];
    }

    const double *CorrelatorsContainer::ra()
    {
        return &fr[0];
    }

    std::vector<double> *CorrelatorsContainer::rxw()
    {
        // frxw.clear(); // no longer needed since vector size is pre-allocated
        for (size_t i = 0; i < fr.size(); i++)
        {
            frxw[i] = fr[i] * fw[i];
        }
        return &frxw;
    }

    void CorrelatorsContainer::PrintLists()
    { // print content of the pair lists
        for (size_t i = 0; i < fr.size(); i++)
        {
            if (findx1.size()==fr.size())
            {
                std::cout << "pair " << i << " distance " << fr[i] << " weight " << fw[i] << " index1 " << findx1[i] << " index2 " << findx2[i] << std::endl;
            }
            else
            {
                std::cout << "pair " << i << " distance " << fr[i] << " weight " << fw[i] << std::endl;
            }
        }
    }

    std::vector<fastjet::PseudoJet> constituents_as_vector(const fastjet::PseudoJet &jet)
    {
        std::vector<fastjet::PseudoJet> _v;
        for (auto &c : jet.constituents())
        {
            _v.push_back(c);
        }
        return _v;
    }

    CorrelatorBuilder::CorrelatorBuilder()
    : fec()
    , fncmax(0)
    {
        ;
    }

    CorrelatorBuilder::CorrelatorBuilder(const std::vector<fastjet::PseudoJet> &parts, const double &scale, const int &nmax, const int &power, const double dphi_cut = -9999, const double deta_cut = -9999)
    : fec(nmax - 1)
    , fncmax(nmax)
    {
        // std::cout << "Initializing n point correlator with power " << power << " for " << parts.size() << " particles" << std::endl;
        int nparts = parts.size();
        int ngroups{0}, idx_i, idx_j, max_idx_i, max_idx_j;
        double _w2, _max_deltaR, _deltaR;
        if (fncmax < 2) {
            throw std::overflow_error("asking for n-point correlator with n < 2?");
        }
        for (int ipoint = 2; ipoint <= fncmax; ipoint++) {
            // need this calculation of ngroups outside creation of CorrelatorsContainer to properly size fidx
            ngroups = calcGroups(nparts, ipoint);

            fec[ipoint - 2] = new CorrelatorsContainer(ipoint, parts.size(), ngroups);
            for (int igroup = 0; igroup < ngroups; igroup++) { // cycle through each group
                _max_deltaR = -1;
                auto idxs_group = fec[ipoint - 2]->indices()->operator[](igroup);
                for (int i = 0; i < ipoint - 1; i++) { // cycle through pairs of each group to find max RL
                    idx_i = idxs_group[i];
                    for (int j = i + 1; j < ipoint; j++) {
                        idx_j = idxs_group[j];
                        // if (ipoint == 2) {
                        //     if (dphi_cut > -1) { // if dphi_cut is on, apply it to pairs
                        //         double _phi12 = fabs(parts[idx_i].delta_phi_to(parts[idx_j])); // delta_phi_to returns [-pi, pi]
                        //         int _q1 = 1; // FIXME: just dummy (no charge info available yet in data and full sim)
                        //         int _q2 = 1;
                        //         if ( !ApplyDeltaPhiRejection(dphi_cut, _q1, _q2, parts[idx_i].pt(), parts[idx_j].pt(), _phi12) ) continue;
                        //     }
                        //     if (deta_cut > -1) { // if deta_cut is on, apply it to pairs
                        //         double _eta12 = parts[idx_i].eta() - parts[idx_j].eta();
                        //         if ( !ApplyDeltaEtaRejection(deta_cut, _eta12) ) continue;
                        //     }
                        // }

                        // _deltaR = parts[idx_i].delta_R(parts[idx_j]);
                        _deltaR = std::sqrt(std::pow(parts[idx_i].delta_phi_to(parts[idx_j]), 2) + std::pow(parts[idx_i].eta() - parts[idx_j].eta(), 2));
                        if (_deltaR > _max_deltaR) {
                            _max_deltaR = _deltaR;
                            max_idx_i = idx_i;
                            max_idx_j = idx_j;
                        }
                    }
                }
                _w2 = 1 / std::pow(scale, ipoint);
                for (auto &grpidx : idxs_group) {
                    _w2 *= parts[grpidx].perp();
                }
                _w2 = pow(_w2, power);
                fec[ipoint - 2]->addwr(igroup, _w2, _max_deltaR, max_idx_i, max_idx_j); // save weight, distance and contributing indices of the pair
            }
        }




        // old code
        // for (size_t i = 0; i < parts.size(); i++)
        // {
        //     for (size_t j = 0; j < parts.size(); j++)
        //     {
        //         double _phi12 = fabs(parts[i].delta_phi_to(parts[j])); // expecting delta_phi_to() to return values in [-pi, pi]
        //         double _eta12 = parts[i].eta() - parts[j].eta();
        //         if (dphi_cut > -1)
        //         { // if dphi_cut is on, apply it to pairs
        //             double _pt1 = parts[i].pt();
        //             double _pt2 = parts[j].pt();
        //             int _q1 = 1; // FIX ME: just dummy (no charge info available yet in data and full sim)
        //             int _q2 = 1;
        //             if ( !ApplyDeltaPhiRejection(dphi_cut, _q1, _q2, _pt1, _pt2, _phi12) ) continue;
        //         }
        //         if (deta_cut > -1)
        //         { // if deta_cut is on, apply it to pairs
        //             if ( !ApplyDeltaEtaRejection(deta_cut, _eta12) ) continue;
        //         }
        //         double _d12 = parts[i].delta_R(parts[j]);
        //         double _w2 = parts[i].perp() * parts[j].perp() / std::pow(scale, 2);
        //         _w2 = pow(_w2, power);
        //         // if (is1part) {
        //         //     std::cout << _w2 << " " << _d12 << " " << i << " " << j << std::endl;
        //         // }
        //         fec[2 - 2]->addwr(_w2, _d12, (double)(i), (double)(j)); // save weight, distance and indices of the pair
        //         if (fncmax < 3)
        //             continue;
        //         for (size_t k = 0; k < parts.size(); k++)
        //         {
        //             double _d13 = parts[i].delta_R(parts[k]);
        //             double _d23 = parts[j].delta_R(parts[k]);
        //             double _w3 = parts[i].perp() * parts[j].perp() * parts[k].perp() / std::pow(scale, 3);
        //             _w3 = pow(_w3, power);
        //             double _d3max = std::max({_d12, _d13, _d23});
        //             if (fabs(_d3max-_d12)<1E-5) fec[3 - 2]->addwr(_w3, _d3max, (double)(i), (double)(j));
        //             if (fabs(_d3max-_d13)<1E-5) fec[3 - 2]->addwr(_w3, _d3max, (double)(i), (double)(k));
        //             if (fabs(_d3max-_d23)<1E-5) fec[3 - 2]->addwr(_w3, _d3max, (double)(j), (double)(k));
        //             if (fncmax < 4)
        //                 continue;
        //             for (size_t l = 0; l < parts.size(); l++)
        //             {
        //                 double _d14 = parts[i].delta_R(parts[l]);
        //                 double _d24 = parts[j].delta_R(parts[l]);
        //                 double _d34 = parts[k].delta_R(parts[l]);
        //                 double _w4 = parts[i].perp() * parts[j].perp() * parts[k].perp() * parts[l].perp() / std::pow(scale, 4);
        //                 _w4 = pow(_w4, power);
        //                 double _d4max = std::max({_d12, _d13, _d23, _d14, _d24, _d34});
        //                 if (fabs(_d4max-_d12)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, (double)(i), (double)(j));
        //                 if (fabs(_d4max-_d13)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, (double)(i), (double)(k));
        //                 if (fabs(_d4max-_d23)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, (double)(j), (double)(k));
        //                 if (fabs(_d4max-_d14)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, (double)(i), (double)(l));
        //                 if (fabs(_d4max-_d24)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, (double)(j), (double)(l));
        //                 if (fabs(_d4max-_d34)<1E-5) fec[4 - 2]->addwr(_w4, _d4max, (double)(k), (double)(l));
        //                 if (fncmax < 5)
        //                     continue;
        //                 for (size_t m = 0; m < parts.size(); m++)
        //                 {
        //                     double _d15 = parts[i].delta_R(parts[m]);
        //                     double _d25 = parts[j].delta_R(parts[m]);
        //                     double _d35 = parts[k].delta_R(parts[m]);
        //                     double _d45 = parts[l].delta_R(parts[m]);
        //                     double _w5 = parts[i].perp() * parts[j].perp() * parts[k].perp() * parts[l].perp() * parts[m].perp() / std::pow(scale, 5);
        //                     _w5 = pow(_w5, power);
        //                     double _d5max = std::max({_d12, _d13, _d23, _d14, _d24, _d34, _d15, _d25, _d35, _d45});
        //                     if (fabs(_d5max-_d12)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(i), (double)(j));
        //                     if (fabs(_d5max-_d13)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(i), (double)(k));
        //                     if (fabs(_d5max-_d23)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(j), (double)(k));
        //                     if (fabs(_d5max-_d14)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(i), (double)(l));
        //                     if (fabs(_d5max-_d24)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(j), (double)(l));
        //                     if (fabs(_d5max-_d34)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(k), (double)(l));
        //                     if (fabs(_d5max-_d15)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(i), (double)(m));
        //                     if (fabs(_d5max-_d25)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(j), (double)(m));
        //                     if (fabs(_d5max-_d35)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(k), (double)(m));
        //                     if (fabs(_d5max-_d35)<1E-5) fec[5 - 2]->addwr(_w5, _d5max, (double)(l), (double)(m));
        //                 }
        //             }
        //         }
        //     }
        // }
    }

    CorrelatorsContainer* CorrelatorBuilder::correlator(int n)
    {
        if (n > fncmax)
        {
            throw std::overflow_error("requesting n-point correlator with too large n");
        }
        if (n < 2)
        {
            throw std::overflow_error("requesting n-point correlator with n < 2?");
        }
        return fec[n - 2];
    }

    CorrelatorBuilder::~CorrelatorBuilder()
    {
        for (auto p : fec)
        {
            delete p;
        }
        fec.clear();
    }

    bool CorrelatorBuilder::ApplyDeltaPhiRejection(const double dphi_cut, const double q1, const double q2, const double pt1, const double pt2, const double phi12)
    {
        double R = 1.1; // reference radius for TPC
        double Bz = 0.5;
        double phi_star = phi12 + q1*asin(-0.015*Bz*R/pt1) - q2*asin(-0.015*Bz*R/pt2);
        if ( fabs(phi_star)<dphi_cut ) return false;  
        return true;
    }

    bool CorrelatorBuilder::ApplyDeltaEtaRejection(const double deta_cut, const double eta12)
    {
        if ( fabs(eta12) < deta_cut ) return false;
        return true;
    }

    int CorrelatorBuilder::calcGroups(int nparts, int ipoint)
    { // pass nparts and ipoint by value and NOT by reference!
        // Counting combinations WITHOUT replacement
        // if (ipoint > nparts) return 0;
        // if (ipoint * 2 > nparts) ipoint = nparts-ipoint;

        // int result = nparts;
        // for( int i = 2; i <= ipoint; ++i ) {
        //     result *= (nparts-i+1);
        //     result /= i;
        // }
        // return result;

        // Counting combinations WITH replacement:
        // if (ipoint * 2 > (nparts + ipoint - 1)) ipoint = nparts - 1;

        // int result = nparts + ipoint - 1;
        // for( int i = 2; i <= ipoint; ++i ) {
        //     result *= (nparts + ipoint - i);
        //     result /= i;
        // }
        // return result;

        // Counting permutations WITH replacement:
        return std::pow(nparts, ipoint);
    }

	std::vector<fastjet::PseudoJet> merge_signal_background_pjvectors(const std::vector<fastjet::PseudoJet> &signal, 
																	  const std::vector<fastjet::PseudoJet> &background,
																      const double pTcut,
																	  const int bg_index_start)
    {
        std::vector<fastjet::PseudoJet> _vreturn;
        auto _selector = fastjet::SelectorPtMin(pTcut);
        auto _signal = _selector(signal);
        for (auto &_p : _signal)
        {
            _p.set_user_index(_p.user_index());
            _vreturn.push_back(_p);
        }
        auto _background = _selector(background);
        for (auto &_p : _background)
        {
            if (bg_index_start > 0)
            {
                int _index = &_p - &_background[0];
                _p.set_user_index(bg_index_start + _index);
            }
            else
            {
                _p.set_user_index(_p.user_index());
            }
            _vreturn.push_back(_p);
        }
        return _vreturn;
    }

}
