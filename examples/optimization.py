import logging
from solver.settings import PetrovGalerkinSolverSettings,LinearSolver
from hybrid.optim import OptimalDesign,NLLS,GradientTest
from hybrid.eqtns import DesignSequences
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def nlls_demo():
    pgs = PetrovGalerkinSolverSettings(degree_increments = (2,4,8,),\
                    max_rep_err=1e-2,max_lcl_err=1e-3,max_num_interval=2**12,)
    linsolv = LinearSolver(pgs,) 
    ds = DesignSequences(n = 1000,random_initialization_seed = 0)
    
    nlls = NLLS(linsolv,driving_functions = ds)
    nlls.set_true_solution(m0s = 0.05,)
    nlls.corrupt_with_gaussian_noise(1e-2)
    logging.info(f'nlls.eval(m0s = 0.2,) = {nlls.eval(m0s = 0.2,)}')
    logging.info(f'nlls.eval(m0s = 0.06,) = {nlls.eval(m0s = 0.06,)}')
    logging.info(f'nlls.eval(m0s = 0.04,) = {nlls.eval(m0s = 0.04,)}')
    logging.info(f'nlls.eval(m0s = 0.045,) = {nlls.eval(m0s = 0.045,)}')
def optdes_demo():
    pgs = PetrovGalerkinSolverSettings(degree_increments = (2,4,8,),\
                    max_rep_err=1e-2,max_lcl_err=1e-3,max_num_interval=2**12,)
    linsolv = LinearSolver(pgs,)
    ds = DesignSequences(n = 50,random_initialization_seed = 0)
    optdes = OptimalDesign(linsolv,driving_functions = ds)
    
    gt = GradientTest(optdes,ds.design_vec,seed = 0)
    loss = optdes.eval(ds.design_vec)
    optdes.jac(ds.design_vec)
    logging.info(f'loss = {loss}')

    
    
    relerr = gt.find_best_match(1/10,9)
    logging.info(f'relerr = {relerr}')
    
    
    fng = optdes.params_fingerprints
    plt.plot(fng.fingerprint_times,fng.values,label = fng.signal_names)
    plt.legend()
    plt.savefig('dummy.png')
    
def main():
    optdes_demo()


if __name__ == '__main__':
    main()