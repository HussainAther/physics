      program generate 
      implicit double precision (a-h,o-z) 
      dimension x(10) 
      common /acc/ nac
      dimension qlab(4),rlabt(4)
      double precision kplabt(4),kmlabt(4),k0labt(4)
      common /input/ a(8) 
      common /flaga/ ifla


      xmp = 0.1395
      xmn = 0.9383
      eg = 8.
      s = 2.*eg*xmn + xmn**2
      call initbins(eg)


c     specify amplitudes for each "amplitude" 
c     i.e. a(1) = 1 -> a2 production x a(1) 
c     a(1) -> a2
c     a(2) -> a1S
c     a(3) -> a2D
c     a(4) -> pi2P
c     a(5) -> pi2F
c     a(6) -> pi2S
c     a(7) -> pi2D
c     a(8) -> piP (exotic) 

c     if a(i) = 0 for all i this will generate phase space 

      do i=1,8
       a(i) = 0.
c       a(i) = 1.
      enddo 
      a(1) = 1.

      if ((a(1).eq.0.).and.(a(2).eq.0.).and.(a(3).eq.0.).and.
     $(a(4).eq.0.).and.(a(5).eq.0.).and.(a(6).eq.0.).and.(a(7).eq.0)
     $ .and.(a(8).eq.0.) ) then
       ifla = 1
       else
       ifla = 2
       endif




c------- first termalize wtih n =1000 steps 

      trial = 5.
      nac = 0
      idum = 12346
      n = 10000
      call markovn(s,idum,n,trial)
      open(10,file='data/MC',status='unknown')
 
c---     enerate nev events 
      nev = 100000
      do iev=1,nev
      if (dble(iev/1000).eq.dble(iev)/1000.) then
      write(*,*) iev
      endif 


      call markovp(s,idum,x,trial)

      call gen(eg,x,qlab,rlabt,kplabt,kmlabt,k0labt,pol,phi_g)

      write(10,*) pol,phi_g
      write(10,*) qlab(1),qlab(2),qlab(3),qlab(4)
      write(10,*) rlabt(1),rlabt(2),rlabt(3),rlabt(4)
      write(10,*) kplabt(1),kplabt(2),kplabt(3),kplabt(4)
      write(10,*) kmlabt(1),kmlabt(2),kmlabt(3),kmlabt(4)
      write(10,*) k0labt(1),k0labt(2),k0labt(3),k0labt(4)


      enddo

      write(*,*) dble(nac)/dble(nev)



      stop
      end 


      subroutine markovn(s,idum,n,trial)
      implicit double precision (a-h,o-z)
      common /mark/ arga(10)
      dimension argb(10) 
      pi = 4.d0*datan(1.d0)
      xmp = 0.1395
      xmn = 0.9383

      xm123min = 3.*xmp
      xm123max = dsqrt(s) - xmn


      tgjmin = -1.d0
      tgjmax =  1.d0
      phimin = 0.d0
      phimax = 2.*pi
      xm12max = dsqrt(s) - xmn - xmp
      xm12min = 2.*xmp

      dp = (phimax-phimin)/trial
      dt = (tgjmax-tgjmin)/trial
      d123 = (xm123max-xm123min)/trial
      d12 = (xm12max-xm12min)/trial
 



c      xm12 = arg(1)
c      xm123 = arg(2)
c      tgj_cm = arg(3)
c      phi_cm = arg(4)
c      tgj_0 = arg(5)
c      phi_0 = arg(6)
c      tgj_t = arg(7) 
c      phi_t = arg(8) 
c     pol = arg(9)
c     phi_g = arg(10) 

c--------------------------------------------
 10   continue
      arga(9) = 1.
      arga(10) = 0.

      a = ran2(idum) 
      arga(7)  = tgjmin*(1.-a) + tgjmax*a
      a = ran2(idum)
      arga(8) = phimin*(1.-a) + phimax*a
      a = ran2(idum) 
      arga(5) = tgjmin*(1.-a) + tgjmax*a
      a = ran2(idum)
      arga(6) = phimin*(1.-a) + phimax*a
      a = ran2(idum) 
      arga(3) = tgjmin*(1.-a) + tgjmax*a
      a = ran2(idum)
      arga(4) = phimin*(1.-a) + phimax*a
      a = ran2(idum) 
      arga(2) = xm123min*(1.-a) + xm123max*a
      a = ran2(idum)
      arga(1) = xm12min*(1.-a) + xm12max*a

      ola = w(s,arga)
      if (ola.eq.0.) goto 10
c------------------------------------

      do iev=1,n

      argb(9) = 1.
      argb(10) = 0.

c      a = ran2(idum) 
c      argb(7)  = tgjmin*(1.-a) + tgjmax*a
c      a = ran2(idum)
c      argb(8) = phimin*(1.-a) + phimax*a
c      a = ran2(idum) 
c      argb(5) = tgjmin*(1.-a) + tgjmax*a
c      a = ran2(idum)
c      argb(6) = phimin*(1.-a) + phimax*a
c      a = ran2(idum) 
c      argb(3) = tgjmin*(1.-a) + tgjmax*a
c      a = ran2(idum)
c      argb(4) = phimin*(1.-a) + phimax*a
c      a = ran2(idum) 
c      argb(2) = xm123min*(1.-a) + xm123max*a
c      a = ran2(idum)
c      argb(1) = xm12min*(1.-a) + xm12max*a


      a = ran2(idum) 
      argb(7) = arga(7) - (dt/2.)*(1.-a) + (dt/2.)*a
      a = ran2(idum)
      argb(8) = arga(8) - (dp/2.)*(1.-a) + (dp/2.)*a
      a = ran2(idum) 
      argb(5) = arga(5) - (dt/2.)*(1.-a) + (dt/2.)*a
      a = ran2(idum)
      argb(6) = arga(6) - (dp/2.)*(1.-a) + (dp/2.)*a
      a = ran2(idum) 
      argb(3) = arga(3) - (dt/2.)*(1.-a) + (dt/2.)*a
      a = ran2(idum)
      argb(4) = arga(4) - (dp/2.)*(1.-a) + (dp/2.)*a
      a = ran2(idum) 
      argb(2) = arga(2) - (d123/2.)*(1.-a) + (d123/2.)*a
      a = ran2(idum)
      argb(1) = arga(1) - (d12/2.)*(1.-a) + (d12/2.)*a

      if ((argb(1).lt.xm12min).or.(argb(1).gt.xm12max).or.
     $    (argb(2).lt.xm123min).or.(argb(2).gt.xm123max).or.
     $   (argb(4).lt.phimin).or.(argb(4).gt.phimax).or.
     $   (argb(3).lt.tgjmin).or.(argb(3).gt.tgjmax).or.
     $   (argb(6).lt.phimin).or.(argb(6).gt.phimax).or.
     $   (argb(5).lt.tgjmin).or.(argb(5).gt.tgjmax).or.
     $   (argb(8).lt.phimin).or.(argb(8).gt.phimax).or.
     $   (argb(7).lt.tgjmin).or.(argb(7).gt.tgjmax) ) then
      r = 0.
      else 
      r = w(s,argb)/w(s,arga) 
      endif 
      if (r.gt.1) then
      do i=1,10
      arga(i) = argb(i)
      enddo 
      else
      b = ran2(idum)
      if (b.le.r) then 
      do i=1,10
      arga(i) = argb(i)
      enddo 
      endif 
      endif


      enddo 

      return
      end 



      subroutine markovp(s,idum,x,trial)
      implicit double precision (a-h,o-z)
      common /mark/ xtmp(10)
      common /acc/ nac
      common /flaga/ ifla
	logical new
      dimension x(10),argb(10)
      pi = 4.d0*datan(1.d0)
      xmp = 0.1395
      xmn = 0.9383

      xm123min = 3.*xmp
      xm123max = dsqrt(s) - xmn

      tgjmin = -1.d0
      tgjmax =  1.d0
      phimin = 0.d0
      phimax = 2.*pi
      xm12max = dsqrt(s) - xmn - xmp
      xm12min = 2.*xmp

      dp = (phimax-phimin)/trial
      dt = (tgjmax-tgjmin)/trial
      d123 = (xm123max-xm123min)/trial
      d12 = (xm12max-xm12min)/trial
 
	new=.false.

      argb(9) = 1.
      argb(10) = 0.

	if (new) 	ifla=1
44	continue
      if (ifla.eq.1) then 
      a = ran2(idum) 
      argb(7)  = tgjmin*(1.-a) + tgjmax*a
      a = ran2(idum)
      argb(8) = phimin*(1.-a) + phimax*a
      a = ran2(idum) 
      argb(5) = tgjmin*(1.-a) + tgjmax*a
      a = ran2(idum)
      argb(6) = phimin*(1.-a) + phimax*a
      a = ran2(idum) 
      argb(3) = tgjmin*(1.-a) + tgjmax*a
      a = ran2(idum)
      argb(4) = phimin*(1.-a) + phimax*a
      a = ran2(idum) 
      argb(2) = xm123min*(1.-a) + xm123max*a
      a = ran2(idum)
      argb(1) = xm12min*(1.-a) + xm12max*a
      else
      a = ran2(idum) 
      argb(7) = xtmp(7) - (dt/2.)*(1.-a) + (dt/2.)*a
      a = ran2(idum)
      argb(8) = xtmp(8) - (dp/2.)*(1.-a) + (dp/2.)*a
      a = ran2(idum) 
      argb(5) = xtmp(5) - (dt/2.)*(1.-a) + (dt/2.)*a
      a = ran2(idum)
      argb(6) = xtmp(6) - (dp/2.)*(1.-a) + (dp/2.)*a
      a = ran2(idum) 
      argb(3) = xtmp(3) - (dt/2.)*(1.-a) + (dt/2.)*a
      a = ran2(idum)
      argb(4) = xtmp(4) - (dp/2.)*(1.-a) + (dp/2.)*a
      a = ran2(idum) 
      argb(2) = xtmp(2) - (d123/2.)*(1.-a) + (d123/2.)*a
      a = ran2(idum)
      argb(1) = xtmp(1) - (d12/2.)*(1.-a) + (d12/2.)*a
      endif 

      if ((argb(1).lt.xm12min).or.(argb(1).gt.xm12max).or.
     $    (argb(2).lt.xm123min).or.(argb(2).gt.xm123max).or.
     $   (argb(4).lt.phimin).or.(argb(4).gt.phimax).or.
     $   (argb(3).lt.tgjmin).or.(argb(3).gt.tgjmax).or.
     $   (argb(6).lt.phimin).or.(argb(6).gt.phimax).or.
     $   (argb(5).lt.tgjmin).or.(argb(5).gt.tgjmax).or.
     $   (argb(8).lt.phimin).or.(argb(8).gt.phimax).or.
     $   (argb(7).lt.tgjmin).or.(argb(7).gt.tgjmax) ) then
      r = 0.
      else
      r = w(s,argb)/w(s,xtmp) 
      endif 



	if(new)	goto 33

      if (r.gt.1) then
       do i=1,10
       xtmp(i) = argb(i)
       x(i) = xtmp(i)
       enddo
       nac = nac + 1
      else
       b = ran2(idum)
       if (b.le.r) then 
        do i=1,10
         xtmp(i) = argb(i)
         x(i) = xtmp(i)
        enddo
        nac = nac + 1
       else
        do i=1,10
c       xtmp(i) = arga(i)
         x(i) = xtmp(i)
        enddo
       endif
      endif 

	return

33     continue
	b=ran2(idum)
      if (r.gt.0.and.r.lt.1.and.b.lt.r) then
        do i=1,10
         x(i) = argb(i)
        enddo
      else
        nac = nac + 1
	goto 44
      endif
	


      return
      end 



 


      function w(s,arg)
      implicit double precision (a-h,o-z)
      double complex zero
      double precision k12_pm,k12_0p,k12_m0
      dimension arg(10),argpm(10),arg0p(10),argm0(10)
      dimension b(20)
      double complex a2(2,2,3),a1S(2,2,3),a1D(2,2,3)
      double complex pi2P(2,2,3),pi2F(2,2,3)
      double complex pi2S(2,2,3),pi2D(2,2,3),pi1P(2,2,3)

      double complex viso(20)
      double complex ampl(2,2,3)


      common /input/ a(8) 

      zero = (0.d0,0.d0) 
      do ig=-1,1,2
      do ln=1,2
      do lnr=1,2
      a2(ln,lnr,ig+1+1) = zero
      a1S(ln,lnr,ig+1+1) = zero
      a1D(ln,lnr,ig+1+1)= zero
      pi2P(ln,lnr,ig+1+1)= zero
      pi2F(ln,lnr,ig+1+1)= zero
      pi2S(ln,lnr,ig+1+1)= zero
      pi2D(ln,lnr,ig+1+1)= zero
      pi1P(ln,lnr,ig+1+1)= zero
      enddo
      enddo
      enddo




      xmp = 0.1395
      xmn = 0.9383
      eg = (s - xmn**2)/(2.*xmn)

      xm123 = arg(2)
      tgj_t = arg(7) 
      phi_t = arg(8) 
      pol = arg(9)


      if (arg(1).ge.(xm123-xmp)) then 
      w = 0.d0
      return 
      endif 


      pp = dsqrt((s-(xm123-xmn)**2)*(s-(xm123+xmn)**2)/(4.*s))
      tt = xm123**4/(4.*s) - pp**2*(1.-tgj_t**2)
     $  - ( tgj_t*pp - (s-xmn**2)/(2.*dsqrt(s)) )**2
      ttmin =  xm123**4/(4.*s) 
     $  - ( pp - (s-xmn**2)/(2.*dsqrt(s)) )**2

      call convert(eg,arg,argpm,arg0p,argm0)
      phi_g = argpm(10) 

      xm12_pm = argpm(1)
      tgj_cm_pm = argpm(3)
      phi_cm_pm = argpm(4)
      tgj_0_pm = argpm(5)
      phi_0_pm = argpm(6)


      k12_pm  = dsqrt((xm123**2-(xm12_pm-xmp)**2)
     $ *(xm123**2-(xm12_pm+xmp)**2)/(4.*xm123**2))
      q12_pm = dsqrt(xm12_pm**2/4. - xmp**2)

      xm12_0p = arg0p(1)
      tgj_cm_0p = arg0p(3)
      phi_cm_0p = arg0p(4)
      tgj_0_0p = arg0p(5)
      phi_0_0p = arg0p(6)


      k12_0p  = dsqrt((xm123**2-(xm12_0p-xmp)**2)
     $ *(xm123**2-(xm12_0p+xmp)**2)/(4.*xm123**2))
      q12_0p = dsqrt(xm12_0p**2/4. - xmp**2)

      xm12_m0 = argm0(1)
      tgj_cm_m0 = argm0(3)
      phi_cm_m0 = argm0(4)
      tgj_0_m0 = argm0(5)
      phi_0_m0 = argm0(6)

      k12_m0  = dsqrt((xm123**2-(xm12_m0-xmp)**2)
     $ *(xm123**2-(xm12_m0+xmp)**2)/(4.*xm123**2))
      q12_m0 = dsqrt(xm12_m0**2/4. - xmp**2)




      if ((xm12_pm.ge.(xm123-xmp)).or.(xm12_0p.ge.(xm123-xmp))
     $ .or.(xm12_m0.ge.(xm123-xmp)) ) then
      w = 0.d0
      return 
      endif 

      if ((a(1).eq.0.).and.(a(2).eq.0.).and.(a(3).eq.0.).and.
     $(a(4).eq.0.).and.(a(5).eq.0.).and.(a(6).eq.0.).and.(a(7).eq.0)
     $ .and.(a(8).eq.0.) ) then 
       w = pp*k12_pm*q12_pm
       return
       endif 

c----   params for a2 -------------
c         DECAY 
      Par = +1.
      J = 2
      L = 2
      iS = 1
      IX = 1
      Ii = 1
      xm_X = 1.318
      gamma_X = 0.105
      br_X = 0.7
      xm_i = 0.775
      gamma_i = 0.146
      br_i = 1.
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 200. 
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1. 

      if (a(1).ne.0.) then 
       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,a2)
      endif 


c-----------end params a2 --------------------

c----   params for a1 (S) -----------------
c         DECAY 
      Par = +1.
      J = 1
      L = 0
      iS = 1
      IX = 1
      Ii = 1
      xm_X = 1.230
      gamma_X = 0.4
      br_X = 0.6*(1./(1. + 0.062**2))
      xm_i = 0.775
      gamma_i = 0.146
      br_i = 1.
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 600. 
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1. 

      if (a(2).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 
       call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,a1S)

       endif 
 

c-----------end params a1 (S) --------------------


c----   params for a1 (D) -----------------
c         DECAY
      Par = +1. 
      J = 1
      L = 2
      iS = 1
      IX = 1
      Ii = 1
      xm_X = 1.230
      gamma_X = 0.4
      br_X = 0.6*(0.062**2/(1. + 0.062**2))
      xm_i = 0.775
      gamma_i = 0.146
      br_i = 1.
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 600.
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1. 

      if (a(3).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,a1D)

      endif 
c-----------end params a1 (D) --------------------


c----   params for pi2 (P) -----------------
c         DECAY 
      Par = -1.
      J =2 
      L = 1
      iS = 1
      IX = 1
      Ii = 1
      xm_X = 1.670
      gamma_X = 0.259
      br_X = 0.3*(1./(1.+0.72**2))
      xm_i = 0.775
      gamma_i = 0.146
      br_i = 1.
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 300. 
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1. 

      if (a(4).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,pi2P)

      endif 
c-----------end params pi2 (P) --------------------


c----   params for pi2 (F) -----------------
c         DECAY 
      Par = -1.
      J =2
      L = 3
      iS = 1
      IX = 1
      Ii = 1
      xm_X = 1.670
      gamma_X = 0.259
      br_X = 0.3*(0.72**2/(1.+0.72**2))
      xm_i = 0.775
      gamma_i = 0.146
      br_i = 1.
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 300.
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1.

      if (a(5).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,pi2F)

      endif 

c-----------end params pi2 (F) --------------------


c----   params for pi2 (S (f2pi)) -----------------
c         DECAY
      Par = -1. 
      J =2
      L = 0 
      iS = 2 
      IX = 1
      Ii = 0
      xm_X = 1.670
      gamma_X = 0.259
      br_X = 0.6*(1./(1. + 0.18**2))
      xm_i = 1.270 
      gamma_i = 0.185
      br_i = 0.85
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 300.
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1.

      if (a(6).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,pi2S)

      endif 

c-----------end params pi2 (S (f2pi)) --------------------


c----   params for pi2 (D (f2pi)) -----------------
c         DECAY
      Par = -1. 
      J =2
      L = 2
      iS = 2
      IX = 1
      Ii = 0
      xm_X = 1.670
      gamma_X = 0.259
      br_X = 0.6*(0.18**2/(1. + 0.18**2))
      xm_i = 1.270
      gamma_i = 0.185
      br_i = 0.85
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 300.
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1.

      if (a(7).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,pi2D)

       endif 

c-----------end params pi2 (D (f2pi)) --------------------


c----   params for pi1 (P) -----------------
c         DECAY 
      Par = -1.
      J = 1 
      L = 1
      iS = 1
      IX = 1
      Ii = 1
      xm_X = 1.600
      gamma_X = 0.200
      br_X = 0.2
      xm_i = 0.775
      gamma_i = 0.146
      br_i = 1.
      sl = 1.
c         PRODUCTION 
c      b(0) no flip, b(1) 1 flip, b(2) 2 flip b(3) 3 flip 
c      bL no flip (nucleon) 
c      bT flip  (nucleon) 
      b(0+1) = 300. 
      b(1+1) = 0.
      b(2+1) = 0.
      b(3+1) = 0.
      bL = 0.
      bT = 1. 

      if (a(8).ne.0.) then 

       call isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 

      call sumampl(J,Par,bl,bt,b,viso,tt,ttmin,pi1P)

      endif 
c-----------end params pi1P (P) --------------------

 66   continue


      do ig=-1,1,2
      do ln=1,2
      do lnr=1,2
      ampl(ln,lnr,ig+1+1) = 
     $       a(1)*a2(ln,lnr,ig+1+1)
     $     + a(2)*a1S(ln,lnr,ig+1+1)
     $     + a(3)*a1D(ln,lnr,ig+1+1)
     $     + a(4)*pi2P(ln,lnr,ig+1+1)
     $     + a(5)*pi2F(ln,lnr,ig+1+1)
     $     + a(6)*pi2S(ln,lnr,ig+1+1)
     $     + a(7)*pi2D(ln,lnr,ig+1+1)
     $     + a(8)*pi1p(ln,lnr,ig+1+1) 


      enddo
      enddo
      enddo 


c       call totalampl(J,bl,bt,b,viso,tt,ttmin,pol,phi_g,ampl)
c       w = pp*ampl*k12_pm*q12_pm

c---    to generate wit amps 
       call photon(pol,phi_g,ampl,tot)
       w = pp*tot*k12_pm*q12_pm

      return
      end 
      


 
