# my_graph.py
from matplotlib.pyplot import close, figure, plot, xlabel, ylabel, title, savefig, show
from numpy import expand_dims,linspace, squeeze
import os

def track_growth(par, sol, sim):
    figure(1)
    plot(expand_dims(par.kgrid,axis=1), squeeze(sol['y'][:, 0, :]))  # Revert to using plot
    xlabel('$k_{t}$')
    ylabel('$y_{t}$')
    title('Production Function')

  #  figname = os.path.join(par.figout, "ypol.png")
  #  savefig(figname)

    figure(2)
    plot(expand_dims(par.kgrid,axis=1), squeeze(sol['k'][:, 0, :])) # Revert to using plot
    xlabel('$k_{t}$')
    ylabel('$k_{t+1}$')
    title('Capital Policy Function')
  #  figname = os.path.join(par.figout, "kpol.png")
  #  savefig(figname)

    figure(3)
    plot(expand_dims(par.kgrid,axis=1), squeeze(sol['c'][:, 0, :])) # Revert to using plot
    xlabel('$k_{t}$')
    ylabel('$c_{t}$')
    title('Consumption Policy Function')
  #  figname = os.path.join(par.figout, "cpol.png")
  #  savefig(figname)

    figure(4)
    plot(expand_dims(par.kgrid,axis=1), squeeze(sol['i'][:, 0, :])) # Revert to using plot
    xlabel('$k_{t}$')
    ylabel('$i_{t}$')
    title('Investment Policy Function')
  #  figname = os.path.join(par.figout, "ipol.png")
  #  savefig(figname)

    figure(5)
    plot(expand_dims(par.kgrid,axis=1), squeeze(sol['u'][:, 0, :]))  # Revert to using plot
    xlabel('$k_{t}$')
    ylabel('$u_{t}$')
    title('Utility Function')
  #  figname = os.path.join(par.figout, "npol.png")
 #   savefig(figname)

    figure(6)
    plot(expand_dims(par.kgrid,axis=1), squeeze(sol['v'][:, 0, :]))  # Revert to using plot
    xlabel('$k_{t}$')
    ylabel('$v_{t}$')
    title('Value Function')
  #  figname = os.path.join(par.figout, "vfun.png")
  #  savefig(figname)

    tgrid = linspace(1, par.T, par.T, dtype=int)

    figure(7)
    plot(tgrid, sim['ysim'])
    xlabel('Time')
    ylabel('$y^{sim}_t$')
    title('Simulated Output')
   # figname = os.path.join(par.figout, "ysim.png")
   # savefig(figname)

    figure(8)
    plot(tgrid, sim['ksim'])
    xlabel('Time')
    ylabel('$k^{sim}_{t+1}$')
    title('Simulated Capital Choice')
   # figname = os.path.join(par.figout, "ksim.png")
   # savefig(figname)

    figure(9)
    plot(tgrid, sim['csim'])
    xlabel('Time')
    ylabel('$c^{sim}_{t}$')
    title('Simulated Consumption')
  #  figname = os.path.join(par.figout, "csim.png")
   # savefig(figname)

    figure(10)
    plot(tgrid, sim['isim'])
    xlabel('Time')
    ylabel('$i^{sim}_{t}$')
    title('Simulated Investment')
  #  figname = os.path.join(par.figout, "isim.png")
  #  savefig(figname)

    figure(11)
    plot(tgrid, sim['usim'])
    xlabel('Time')
    ylabel('$u^{sim}_t$')
    title('Simulated Utility')
   # figname = os.path.join(par.figout, "usim.png")
   # savefig(figname)

    figure(12)
    plot(tgrid, sim['Asim'])
    xlabel('Time')
    ylabel('$A^{sim}_t$')
    title('Simulated Productivity')
    #figname = os.path.join(par.figout, "Asim.png")
   # savefig(figname)

    figure(13)
    plot(tgrid, sim['n_sim'])
    xlabel('Time')
    ylabel('$n^{sim}_t$')
    title('Simulated Labor Supply')
    #figname = os.path.join(par.figout, "nsim.png")
    #savefig(figname)

    show()
    close('all')
