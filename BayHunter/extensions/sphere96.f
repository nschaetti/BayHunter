c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
        subroutine sphere(ifunc,iflag,d,a,b,rho,rtp,dtp,btp,mmax,llw,
     & twopi)
c-----
c     Transform spherical earth to flat earth
c
c     Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
c     mode computations, in  Methods in Computational Physics,
c         Volume 11,
c     Seismology: Surface Waves and Earth Oscillations,
c         B. A. Bolt (ed),
c     Academic Press, New York
c
c     Love Wave Equations  44, 45 , 41 pp 112-113
c     Rayleigh Wave Equations 102, 108, 109 pp 142, 144
c
c     Revised 28 DEC 2007 to use mid-point, assume linear variation in
c     slowness instead of using average velocity for the layer
c     Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
c
c     ifunc   I*4 1 - Love Wave
c                 2 - Rayleigh Wave
c     iflag   I*4 0 - Initialize
c                 1 - Make model  for Love or Rayleigh Wave
c-----
        parameter(NL=100,NP=60)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        integer mmax,llw
c        common/modl/ d,a,b,rho,rtp,dtp,btp
c        common/para/ mmax,llw,twopi
        double precision twopi
        double precision z0,z1,r0,r1,dr,ar,tmp
        save dhalf
c!f2py name: sphere
c!f2py intent(out) rtp, dtp, btp
        ar=6370.0d0
        dr=0.0d0
        r0=ar
        d(mmax)=1.0
        if(iflag.eq.0) then
            do 5 i=1,mmax
                dtp(i)=d(i)
                rtp(i)=rho(i)
    5       continue
            do 10 i=1,mmax
                dr=dr+dble(d(i))
                r1=ar-dr
                z0=ar*dlog(ar/r0)
                z1=ar*dlog(ar/r1)
                d(i)=z1-z0
c-----
c               use layer midpoint
c-----
                TMP=(ar+ar)/(r0+r1)
                a(i)=a(i)*tmp
                b(i)=b(i)*tmp
                btp(i)=tmp
                r0=r1
   10       continue
            dhalf = d(mmax)
        else
            d(mmax) = dhalf
            do 30 i=1,mmax
                if(ifunc.eq.1)then
                     rho(i)=rtp(i)*btp(i)**(-5)
                else if(ifunc.eq.2)then
                     rho(i)=rtp(i)*btp(i)**(-2.275)
                endif
   30       continue
        endif
        d(mmax)=0.0
        return
        end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c