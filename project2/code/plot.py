import infoFile_ala_Nanna as Nanna

Nanna.sayhello()
Nanna.define_categories({'method':'method', 'opt':'optimiser', 'n_obs':r'$n_\mathrm{obs}$', 'no_epochs':'#epochs', 'eta':r'$\eta$', 'gamma':r'$\gamma$', 'rho':r'$\varrho_1$, $\varrho_2$'})


Nanna.set_file_info("dummy1.jpg", note="helo", eta='0')
#Nanna.omit_file("dummy1.jpg")
#Nanna.omit_file("dummy2.jpg")
Nanna.set_file_info("dummy2.jpg", method='test', rho=1)

Nanna.additional_information("hello")
Nanna.update()