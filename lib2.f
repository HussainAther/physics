
      subroutine photon(pol,phi_g,ampl,tot)
      implicit double precision (a-h,o-z)
      double complex xr,xi
      double complex rho(3,3),ampl(2,2,3)
      xr = (1.d0,0.d0)
      xi = (0.d0,1.d0)
      xmn = 0.9383

      do ii=1,3
      do ki=1,3
      rho(ii,ki) = (0.d0,0.d0)
      enddo
      enddo 

c     circular rho(up,up) = 1 all ather 0 or rho(down,down) = 1 all other 0
c     linear rho(u,u) = 1/2, rho(d,d) =1/2
c            rho(u,d) = -(1/2)*[cos(2phi) - i sin(2phi)]
c            rho(d,u) = -(1/2)*[cos(2phi) + i sin(2ph) ] 

      rho(+1+1+1,+1+1+1)=xr/2.
      rho(-1+1+1,-1+1+1)=xr/2.
      rho(+1+1+1,-1+1+1)=-pol*(dcos(2.*phi_g)*xr - dsin(2.*phi_g)*xi)/2.
      rho(-1+1+1,+1+1+1)=-pol*(dcos(2.*phi_g)*xr + dsin(2.*phi_g)*xi)/2. 

      tot = 0.d0 
      do ig=-1,1,2
      do igp=-1,1,2
      do ln=1,2
      do lnr=1,2

      tot  = tot  + dble( 
     $ ampl(ln,lnr,ig+1+1)*dconjg(ampl(ln,lnr,igp+1+1))
     $   *rho(ig+1+1,igp+1+1)
     $                  ) 


      enddo
      enddo 
      enddo
      enddo 
     
      return
      end 





      subroutine sumampl(J,Par,bl,bt,b,viso,tt,tmin,ampl)
      implicit double precision (a-h,o-z)
      double complex xr,xi
      dimension VU(20,20),VL(2,2)
      dimension b(20)
      double complex viso(20)
      double complex ampl(2,2,3)
      xr = (1.d0,0.d0)
      xi = (0.d0,1.d0)
      xmn = 0.9383

      

      do ii=1,20
      do ki=1,20
      VU(ii,ki) = 0.d0
      enddo
      enddo 

      do ig=-1,1,2
      do ln=1,2
      do lnr=1,2
      ampl(ln,lnr,ig+1+1) = (0.d0,0.d0)
      enddo
      enddo
      enddo 


      ishift = 1
      do igamma=-1,1,2
      do lambda_X=-J,J
      if ((lambda_X.eq.0).and.(igamma.eq.1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(1+ishift)
      endif
      if ((lambda_X.eq.0).and.(igamma.eq.-1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(1+ishift)*Par*(-1.)**J
      endif
      if ((lambda_X.gt.0).and.(igamma.eq.-1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)+1+ishift)
     $ *Par*(-1.)**(J-iabs(lambda_X))
      endif 
      if ((lambda_X.lt.0).and.(igamma.eq.-1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)-1+ishift)
     $  *Par*(-1.)**(J+iabs(lambda_X))
      endif
      if ((lambda_X.lt.0).and.(igamma.eq.1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)+1+ishift)
      endif
      if ((lambda_X.gt.0).and.(igamma.eq.1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)-1+ishift)
      endif 
      enddo
      enddo 



      VL(1,1) = bL
      VL(2,2) = - bL
      VL(1,2) = bt
      VL(2,1) = bt 


      do ig=-1,1,2
      do ln=1,2
      do lnr=1,2
      ampl(ln,lnr,ig+1+1) = (0.d0,0.d0)
      do l_X=-J,J
      if ((ln.eq.1).and.(lnr.eq.1)) then
      xlam = 0.d0
      endif
      if ((ln.eq.2).and.(lnr.eq.2)) then
      xlam = 0.d0
      endif
      if ((ln.eq.1).and.(lnr.eq.2)) then
      xlam = 1./2.
      endif
      if ((ln.eq.2).and.(lnr.eq.1)) then
      xlam = 1./2.
      endif
      exp1 = dble(iabs(ig-l_X))/2. + xlam
      ampl(ln,lnr,ig+1+1) =  ampl(ln,lnr,ig+1+1)
     $ + VL(ln,lnr)*VU(ig+1+1,l_X+J+1)
c     $ *dexp((tt-tmin)/10.)*(- (tt-tmin)/(4.*xmn**2) )**exp1
     $ *(dexp((tt-tmin)*5.))**exp1
     $ *viso(l_X+J+1)
      enddo
      enddo 
      enddo
      enddo 
 
      return
      end 




      subroutine totalampl(J,bl,bt,b,viso,tt,tmin,pol,phi_g,ampl)
      implicit double precision (a-h,o-z)
      double complex xr,xi
      double complex rho(3,3)
      dimension VU(20,20),VL(2,2)
      dimension b(20)
      double complex viso(20)
      xr = (1.d0,0.d0)
      xi = (0.d0,1.d0)
      xmn = 0.9383

      do ii=1,20
      do ki=1,20
      VU(ii,ki) = 0.d0
      enddo
      enddo 

      do ii=1,3
      do ki=1,3
      rho(ii,ki) = (0.d0,0.d0)
      enddo
      enddo 



      ishift = 1
      do igamma=-1,1,2
      do lambda_X=-J,J
      if (lambda_X.eq.0) then
      VU(igamma+1+1,lambda_X+J+1) = b(1+ishift)
      endif
      if ((lambda_X.gt.0).and.(igamma.eq.-1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)+1+ishift)
      endif 
      if ((lambda_X.lt.0).and.(igamma.eq.-1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)-1+ishift)
      endif
      if ((lambda_X.lt.0).and.(igamma.eq.1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)+1+ishift)
      endif
      if ((lambda_X.gt.0).and.(igamma.eq.1)) then
      VU(igamma+1+1,lambda_X+J+1) = b(iabs(lambda_X)-1+ishift)
      endif 
      enddo
      enddo 

      VL(1,1) = bL
      VL(2,2) = - bL
      VL(1,2) = bt
      VL(2,1) = bt 



c     circular rho(up,up) = 1 all ather 0 or rho(down,down) = 1 all other 0
c     linear rho(u,u) = 1/2, rho(d,d) =1/2
c            rho(u,d) = -(1/2)*[cos(2phi) - i sin(2phi)]
c            rho(d,u) = -(1/2)*[cos(2phi) + i sin(2ph) ] 

      rho(+1+1+1,+1+1+1)=xr/2.
      rho(-1+1+1,-1+1+1)=xr/2.
      rho(+1+1+1,-1+1+1)=-pol*(dcos(2.*phi_g)*xr - dsin(2.*phi_g)*xi)/2.
      rho(-1+1+1,+1+1+1)=-pol*(dcos(2.*phi_g)*xr + dsin(2.*phi_g)*xi)/2. 
      ampl = 0.d0 
      do ig=-1,1,2
      do igp=-1,1,2
      do ln=1,2
      do lnr=1,2
      do l_X=-J,J
      do lp_X=-J,J
      ala =   VL(ln,lnr)*VU(ig+1+1,l_X+J+1)
      ola =   VL(ln,lnr)*VU(igp+1+1,lp_X+J+1)

      if ((ala.eq.0.).or.(ola.eq.0.)) goto 44


      if ((ln.eq.1).and.(lnr.eq.1)) then
      xlam = 0.d0
      endif
      if ((ln.eq.2).and.(lnr.eq.2)) then
      xlam = 0.d0
      endif
      if ((ln.eq.1).and.(lnr.eq.2)) then
      xlam = 1./2.
      endif
      if ((ln.eq.2).and.(lnr.eq.1)) then
      xlam = 1./2.
      endif


      exp1 = dble(iabs(ig-l_X))/2. + xlam
      exp2 = dble(iabs(igp-lp_X))/2. + xlam


      ampl = ampl + dble( 
     $ ala*ola*dexp(tt-tmin)*(- (tt-tmin)/(4.*xmn**2) )**(exp1+exp2)
     $ *viso(l_X+J+1)*dconjg(viso(lp_X+J+1))*rho(ig+1+1,igp+1+1)
     $                  ) 


 44   continue 
      enddo
      enddo
      enddo
      enddo 
      enddo
      enddo 
     
      return
      end 


      double complex function BW(gamma_X,xm_X,br_X,gamma_i,xm_i,br_i,
     $ xm123,xm12,k12,q12,sL,L,iS)
      implicit double precision (a-h,o-z)
      double precision k12,k120
      double complex xr,xi
      double complex den
      xr = (1.d0,0.d0)
      xi = (0.d0,1.d0)
      xmp = 0.1395

      q0 = dsqrt((xm_X**2-(xm_i-xmp)**2)
     $          *(xm_X**2-(xm_i+xmp)**2)/(4.*xm_X**2))
      gamma = (gamma_X*xm_X)*((k12/xm123)/(q0/xm_X))
     $ *(k12**(2*L)/(xm123**2+sl)**L)/(q0**(2*L)/(xm_X**2+sl)**L)
      xnum = dsqrt(gamma*br_X)
      den = (xm_X**2 - xm123**2 - gamma*xi)
      q0 = dsqrt(xm_i**2/4.-xmp**2)
      gamma = (gamma_i*xm_i)*((q12/xm12)/(q0/xm_i))
     $ *(q12**(2*iS)/(xm12**2+sl)**iS)/(q0**(2*iS)/(xm_i**2+sl)**iS)
      xnum = xnum*dsqrt(gamma*br_i)
      den = den*(xm_i**2 - xm12**2 - gamma*xi)
      BW = xnum/den

c      k120 = dsqrt((xm123**2-(xm_i-xmp)**2)
c     $            *(xm123**2-(xm_i+xmp)**2)/(4.*xm123**2))
c      q0 = dsqrt((xm_X**2-(xm_i-xmp)**2)
c     $          *(xm_X**2-(xm_i+xmp)**2)/(4.*xm_X**2))
c      gamma = (gamma_X*xm_X)*((k120/xm123)/(q0/xm_X))
c     $ *(k120**(2*L)/(xm123**2+sl)**L)/(q0**(2*L)/(xm_X**2+sl)**L)
c      gamma = gamma_X*xm_X
c      den = (xm_X**2 - xm123**2 - gamma*xi)
c      q0 = dsqrt(xm_i**2/4.-xmp**2)
c      gamma = (gamma_i*xm_i)*((q12/xm12)/(q0/xm_i))*(q12/q0)**(2*iS)
c     $ *(q12**(2*iS)/(xm12**2+sl)**iS)/(q0**(2*iS)/(xm_i**2+sl)**iS)
c-------------------------
c      gamma = gamma_X*xm_X
c      den = (xm_X**2 - xm123**2 - gamma*xi)
c      gamma = gamma_i*xm_i
c      den = den*(xm_i**2 - xm12**2 - gamma*xi)
c      BW = dsqrt(gamma_X*xm_X*br_X)*dsqrt(gamma_i*xm_i*br_i)/den


      return
      end 



      subroutine isobar(xm123,
     $ J,L,iS,IX,Ii,xm_X,gamma_X,br_X,xm_i,gamma_i,br_i,sl,
     $ k12_pm,q12_pm,xm12_pm,tgj_0_pm,phi_0_pm,tgj_cm_pm,phi_cm_pm,
     $ k12_0p,q12_0p,xm12_0p,tgj_0_0p,phi_0_0p,tgj_cm_0p,phi_cm_0p,
     $k12_m0,q12_m0,xm12_m0,tgj_0_m0,phi_0_m0,tgj_cm_m0,phi_cm_m0,viso) 
      implicit double precision (a-h,o-z)
      double complex xr,xi
      double complex Y,BW,BW_pm,BW_0p,BW_m0
      double precision k12_pm,k12_0p,k12_m0
      double complex ala,ola,ela 
      double complex viso(20)
      xr = (1.d0,0.d0)
      xi = (0.d0,1.d0)
      xmp = 0.1395
     
      do i=1,20
      viso(i) = (0.d0,0.d0)
      enddo

      BW_pm = BW(gamma_X,xm_X,br_X,gamma_i,xm_i,br_i,
     $ xm123,xm12_pm,k12_pm,q12_pm,sL,L,iS)


      BW_0p = BW(gamma_X,xm_X,br_X,gamma_i,xm_i,br_i,
     $ xm123,xm12_0p,k12_0p,q12_0p,sL,L,iS)


      BW_m0 = BW(gamma_X,xm_X,br_X,gamma_i,xm_i,br_i,
     $ xm123,xm12_m0,k12_m0,q12_m0,sL,L,iS)


      do lambda_X=-J,J
      viso(lambda_X+J+1) = (0.d0,0.d0)
      do ml=-L,L
      do lambda_i=-iS,iS

c    + - 0  --> + + -
     
c     pm -> + + 
      ala = 
     $+ Y(L,ml,tgj_0_pm,phi_0_pm)*Y(iS,lambda_i,tgj_cm_pm,phi_cm_pm)
     $ *cg(iS,lambda_i,L,ml,J,lambda_X)
     $ *BW_pm
c     $ *k12_pm**L*q12_pm**iS
     $ *cg(1,+1,1,+1,Ii,+2)*cg(Ii,+2,1,-1,IX,+1)


     
c     0 + -> - +  (- + +)
      ola = 
     $+ Y(L,ml,tgj_0_0p,phi_0_0p)*Y(iS,lambda_i,tgj_cm_0p,phi_cm_0p)
     $ *cg(iS,lambda_i,L,ml,J,lambda_X)
     $ *BW_0p
c     $*k12_0p**L*q12_0p**iS
     $ *cg(1,-1,1,+1,Ii,0)*cg(Ii,0,1,+1,IX,+1)
     $   



c     - 0 --> + -   (+ - +)
      ela = 
     $+ Y(L,ml,tgj_0_m0,phi_0_m0)*Y(iS,lambda_i,tgj_cm_m0,phi_cm_m0)
     $ *cg(iS,lambda_i,L,ml,J,lambda_X)
     $ *BW_m0
c     $*k12_m0**L*q12_m0**iS
     $ *cg(1,+1,1,-1,Ii,0)*cg(Ii,0,1,+1,IX,+1)

      viso(lambda_X+J+1) = viso(lambda_X+J+1) + ola + ela 
c     $ + ala 
     

      enddo
      enddo
      enddo 

      return
      end 





      subroutine convert(eg,arg,argpm,arg0p,argm0)
      implicit double precision (a-h,o-z)
      dimension arg(10),argpm(10),arg0p(10),argm0(10)
      dimension q(4),r(4)
      double precision kp(4),km(4),k0(4)
      call gen(eg,arg,q,r,kp,km,k0,pol,phi_g)
      call angles(argpm,q,r,kp,km,k0,pol,phi_g)
      call angles(arg0p,q,r,k0,kp,km,pol,phi_g)
      call angles(argm0,q,r,km,k0,kp,pol,phi_g)
      return
      end 


      subroutine gen(eg,arg,qlab,rlabt,kplabt,kmlabt,k0labt,pol,phi_g)
c     input arg (polg_g w.w.t to the lab) 
c     output 4-vectors  and pol,phi_g (phi_g still in the lab) 
      implicit double precision (a-h,o-z)
      dimension arg(10) 
      dimension q(4),p(4),r(4),PP(4)
      double precision kp(4),km(4),kp123(4),km123(4),k0123(4)
      double precision kpcm(4),kmcm(4),k0cm(4)
      double precision kplab(4),kmlab(4),k0lab(4)
      dimension rlab(4),qlab(4),plab(4)
      dimension tmp(4),tmp1(4),tmp2(4)
      dimension rlabt(4)
      double precision  kplabt(4),kmlabt(4),k0labt(4)

      pi = 4.d0*datan(1.d0)
      xmp = 0.1395
      xmn = 0.9383
      s = 2.*eg*xmn + xmn**2      
      xm12 = arg(1)
      xm123 = arg(2)
      tgj_cm = arg(3)
      phi_cm = arg(4)
      tgj_0 = arg(5)
      phi_0 = arg(6)
      tgj_t = arg(7) 
      phi_t = arg(8) 
      pol = arg(9)

      phi_g = arg(10) 
c----------------------
      tgj = tgj_cm
      phi = phi_cm

      kp(1) = dsqrt(1.-tgj**2)*dcos(phi)*dsqrt(xm12**2/4.-xmp**2)
      kp(2) = dsqrt(1.-tgj**2)*dsin(phi)*dsqrt(xm12**2/4.-xmp**2)
      kp(3) = tgj*dsqrt(xm12**2/4.-xmp**2)
      kp(4) = xm12/2. 

      km(1) = -dsqrt(1.-tgj**2)*dcos(phi)*dsqrt(xm12**2/4.-xmp**2)
      km(2) = -dsqrt(1.-tgj**2)*dsin(phi)*dsqrt(xm12**2/4.-xmp**2)
      km(3) = -tgj*dsqrt(xm12**2/4.-xmp**2)
      km(4) = xm12/2.

      tgj = tgj_0
      phi = phi_0

      k0123(1) = -dsqrt(1.-tgj**2)*dcos(phi)
     $*dsqrt((xm123**2-(xm12-xmp)**2)*(xm123**2-(xm12+xmp)**2)
     $ /(4.*xm123**2))
      k0123(2) = -dsqrt(1.-tgj**2)*dsin(phi)
     $*dsqrt((xm123**2-(xm12-xmp)**2)*(xm123**2-(xm12+xmp)**2)
     $ /(4.*xm123**2))
      k0123(3) = -tgj
     $*dsqrt((xm123**2-(xm12-xmp)**2)*(xm123**2-(xm12+xmp)**2)
     $ /(4.*xm123**2))
      k0123(4)=dsqrt(xmp**2+k0123(1)**2+k0123(2)**2+k0123(3)**2)



      PP(1) = -k0123(1)
      PP(2) = -k0123(2)
      PP(3) = -k0123(3)
      PP(4) = dsqrt(xm12**2+PP(1)**2+PP(2)**2+PP(3)**2)
      call boost(kp,kp123,PP,+1.d0)
      call boost(km,km123,PP,+1.d0)



      PP(1) = 0.d0
      PP(2) = 0.d0
      PP(3) = dsqrt((s-(xm123-xmn)**2)*(s-(xm123+xmn)**2)/(4.*s))
      PP(4) = dsqrt(xm123**2+PP(1)**2+PP(2)**2+PP(3)**2)
      call boost(kp123,tmp,PP,+1.d0)
      call boost(km123,tmp1,PP,+1.d0)
      call boost(k0123,tmp2,PP,+1.d0)
c     rotate 

      tgj = tgj_t


      kpcm(1) = tgj*tmp(1) + dsqrt(1.-tgj**2)*tmp(3)
      kpcm(2) = tmp(2)
      kpcm(3) = -dsqrt(1.-tgj**2)*tmp(1) + tgj*tmp(3)
      kpcm(4) = tmp(4)


      kmcm(1) = tgj*tmp1(1) + dsqrt(1.-tgj**2)*tmp1(3)
      kmcm(2) = tmp1(2)
      kmcm(3) = -dsqrt(1.-tgj**2)*tmp1(1) + tgj*tmp1(3)
      kmcm(4) = tmp1(4)


      k0cm(1) = tgj*tmp2(1) + dsqrt(1.-tgj**2)*tmp2(3)
      k0cm(2) = tmp2(2)
      k0cm(3) = -dsqrt(1.-tgj**2)*tmp2(1) + tgj*tmp2(3)
      k0cm(4) = tmp2(4)


      r(1) = -dsqrt(1.-tgj**2)
     $ *dsqrt((s-(xm123-xmn)**2)*(s-(xm123+xmn)**2)/(4.*s))
      r(2) = 0.d0
      r(3) = -tgj
     $ *dsqrt((s-(xm123-xmn)**2)*(s-(xm123+xmn)**2)/(4.*s))
      r(4) = dsqrt(xmn**2+r(1)**2+r(2)**2+r(3)**2)

   
      q(4) = (s-xmn**2)/(2.*dsqrt(s))
      q(1) = 0.d0
      q(2) = 0.d0
      q(3) = (s-xmn**2)/(2.*dsqrt(s))

      p(4) = (s+xmn**2)/(2.*dsqrt(s))
      p(1) = 0.d0
      p(2) = 0.d0
      p(3) = -(s-xmn**2)/(2.*dsqrt(s))



      call boost(p,plab,p,-1.d0)
      call boost(q,qlab,p,-1.d0)
      call boost(r,rlab,p,-1.d0)
      call boost(kpcm,kplab,p,-1.d0)
      call boost(kmcm,kmlab,p,-1.d0)
      call boost(k0cm,k0lab,p,-1.d0)

      phi = phi_t 


      rlabt(1) = rlab(1)*dcos(phi) - rlab(2)*dsin(phi)
      rlabt(2) = rlab(2)*dcos(phi) + rlab(1)*dsin(phi)
      rlabt(3) = rlab(3)
      rlabt(4) = rlab(4)

      kplabt(1) = kplab(1)*dcos(phi) - kplab(2)*dsin(phi)
      kplabt(2) = kplab(2)*dcos(phi) + kplab(1)*dsin(phi)
      kplabt(3) = kplab(3)
      kplabt(4) = kplab(4)

      kmlabt(1) = kmlab(1)*dcos(phi) - kmlab(2)*dsin(phi)
      kmlabt(2) = kmlab(2)*dcos(phi) + kmlab(1)*dsin(phi)
      kmlabt(3) = kmlab(3)
      kmlabt(4) = kmlab(4)

      k0labt(1) = k0lab(1)*dcos(phi) - k0lab(2)*dsin(phi)
      k0labt(2) = k0lab(2)*dcos(phi) + k0lab(1)*dsin(phi)
      k0labt(3) = k0lab(3)
      k0labt(4) = k0lab(4)

c     if phi_g is w.r.t production plane uncomment 
c      phi_g = phi_g - phi_t 


      return 
      end




      subroutine angles(arg,qlab,rlabt,kplabt,kmlabt,k0labt,pol,phi_g)
c     input 4-vectors and pol,phi_g (photon polarization in the lab
c     output angles and pol and phi_g w,r.t to production plane 
      implicit double precision (a-h,o-z)
      dimension arg(10) 
      dimension q(4),p(4),r(4),PP(4),pcm(4),qcm(4)
      double precision kp(4),km(4),kp123(4),km123(4),k0123(4)
      double precision kpcm(4),kmcm(4),k0cm(4)
      double precision kplab(4),kmlab(4),k0lab(4)
      dimension rlab(4),qlab(4),plab(4),rcm(4)
      dimension tmp(4),tmp1(4),tmp2(4),tmp3(4)
      dimension rlabt(4)
      double precision  kplabt(4),kmlabt(4),k0labt(4)
      dimension plane(4)
      common /plane/ planeangle
      dimension ex(3),ey(3),ez(3)

      pi = 4.d0*datan(1.d0)
      xmp = 0.1395
      xmn = 0.9383

      arg(9) = pol

      call add(kplabt,kmlabt,tmp)
      call add(tmp,k0labt,tmp1)


      co2a = tmp1(1)/dsqrt(tmp1(1)**2+tmp1(2)**2)
      si2a = tmp1(2)/dsqrt(tmp1(1)**2+tmp1(2)**2)
      if ((co2a.eq.0.).and.(si2a.eq.0.)) then
      epsf = 0.d0
      else
      if (si2a.ge.0) then
      epsf = dacos(co2a/dsqrt(co2a**2+si2a**2))
      else
      epsf = pi+dacos(-co2a/dsqrt(co2a**2+si2a**2))
      endif
      endif
      phi_t = epsf
c     phi_g is w.r.t lab 
c     arg(10) is w.r.t production plane 
      arg(10) = phi_g - phi_t
c      arg(10) = 0.d0

      call add(kplabt,kmlabt,tmp)
      xm12 = dsqrt(tmp(4)**2-tmp(1)**2-tmp(2)**2-tmp(3)**2)
      call add(tmp,k0labt,tmp1)
      xm123 = dsqrt(tmp1(4)**2-tmp1(1)**2-tmp1(2)**2-tmp1(3)**2) 
      call add(tmp1,rlabt,tmp)
      s = tmp(4)**2-tmp(1)**2-tmp(2)**2-tmp(3)**2


      plab(1) = 0.
      plab(2) = 0.
      plab(3) = 0.
      plab(4) = xmn   
      
      xnor = dsqrt(qlab(1)**2+qlab(2)**2+qlab(3)**2)
      do i=1,3
      ez(i) = qlab(i)/xnor
      enddo

      call add(kplabt,kmlabt,tmp)
      call add(tmp,k0labt,tmp1)
      ey(1) = rlabt(2)*tmp1(3)-rlabt(3)*tmp1(2)
      ey(3) = rlabt(1)*tmp1(2)-rlabt(2)*tmp1(1)
      ey(2) = rlabt(3)*tmp1(1)-rlabt(1)*tmp1(3)
      xnor = dsqrt(ey(1)**2+ey(2)**2+ey(3)**2)
      do i=1,3
      ey(i) = ey(i)/xnor
      enddo
      ex(1) = ey(2)*ez(3)-ey(3)*ez(2)
      ex(3) = ey(1)*ez(2)-ey(2)*ez(1)
      ex(2) = ey(3)*ez(1)-ey(1)*ez(3)

      call project(rlabt,rlab,ex,ey,ez)
      call project(kplabt,kplab,ex,ey,ez)
      call project(kmlabt,kmlab,ex,ey,ez)
      call project(k0labt,k0lab,ex,ey,ez)


      PP(1) = 0.d0
      PP(2) = 0.d0
      PP(3) = -(s-xmn**2)/(2.*dsqrt(s))
      PP(4) =  dsqrt(xmn**2+PP(1)**2+PP(2)**2+PP(3)**2)



      call boost(plab,pcm,PP,1.d0)

      call boost(qlab,qcm,PP,1.d0)

      call boost(rlab,rcm,PP,1.d0)
      call boost(kplab,kpcm,PP,1.d0)
      call boost(kmlab,kmcm,PP,1.d0)
      call boost(k0lab,k0cm,PP,1.d0)



      tgj_t = -rcm(3)/dsqrt(rcm(1)**2+rcm(2)**2+rcm(3)**2)
      tgj = tgj_t



c     rotate

      tmp(1) = tgj*kpcm(1) - dsqrt(1.-tgj**2)*kpcm(3)
      tmp(2) = kpcm(2)
      tmp(3) = +dsqrt(1.-tgj**2)*kpcm(1) + tgj*kpcm(3)
      tmp(4) = kpcm(4)


      tmp1(1) = tgj*kmcm(1) - dsqrt(1.-tgj**2)*kmcm(3)
      tmp1(2) = kmcm(2)
      tmp1(3) = +dsqrt(1.-tgj**2)*kmcm(1) + tgj*kmcm(3)
      tmp1(4) = kmcm(4)


      tmp2(1) = tgj*k0cm(1) - dsqrt(1.-tgj**2)*k0cm(3)
      tmp2(2) = k0cm(2)
      tmp2(3) = +dsqrt(1.-tgj**2)*k0cm(1) + tgj*k0cm(3)
      tmp2(4) = k0cm(4)


      tmp3(1) = tgj*rcm(1) - dsqrt(1.-tgj**2)*rcm(3)
      tmp3(2) = rcm(2)
      tmp3(3) = +dsqrt(1.-tgj**2)*rcm(1) + tgj*rcm(3)
      tmp3(4) = rcm(4)



c     now r is along -z 


      PP(1) = 0.d0
      PP(2) = 0.d0
      PP(3) = dsqrt((s-(xm123-xmn)**2)*(s-(xm123+xmn)**2)/(4.*s))
      PP(4) = dsqrt(xm123**2+PP(1)**2+PP(2)**2+PP(3)**2)


      call boost(tmp,kp123,PP,-1.d0)
      call boost(tmp1,km123,PP,-1.d0)
      call boost(tmp2,k0123,PP,-1.d0)


      tgj_0 = -k0123(3)/dsqrt(k0123(1)**2+k0123(2)**2+k0123(3)**2)
      co2a = -k0123(1)
     $ /(dsqrt(1.-tgj_0**2)*dsqrt(k0123(1)**2+k0123(2)**2+k0123(3)**2))
      si2a = -k0123(2)
     $ /(dsqrt(1.-tgj_0**2)*dsqrt(k0123(1)**2+k0123(2)**2+k0123(3)**2))
      if ((co2a.eq.0.).and.(si2a.eq.0.)) then
      epsf = 0.d0
      else
      if (si2a.ge.0) then
      epsf = dacos(co2a/dsqrt(co2a**2+si2a**2))
      else
      epsf = pi+dacos(-co2a/dsqrt(co2a**2+si2a**2))
      endif
      endif
      phi_0 = epsf
      tgj = tgj_0
      phi = phi_0


cc     also define orientation of the 3pi production plane w.r.t to 
c      production plane in 3pi helicity frame 

      plane(1) = kp123(2)*km123(3)-kp123(3)*km123(2)
      plane(3) = kp123(1)*km123(2)-kp123(2)*km123(1)
      plane(2) = kp123(3)*km123(1)-kp123(1)*km123(3)
      xnor = dsqrt(plane(1)**2+plane(2)**2+plane(3)**2)
      tp  = plane(3)/xnor
      co2a = plane(1)/(dsqrt(1.-tp**2)*xnor)
      si2a = plane(2)/(dsqrt(1.-tp**2)*xnor)
      if ((co2a.eq.0.).and.(si2a.eq.0.)) then
      epsf = 0.d0
      else
      if (si2a.ge.0) then
      epsf = dacos(co2a/dsqrt(co2a**2+si2a**2))
      else
      epsf = pi+dacos(-co2a/dsqrt(co2a**2+si2a**2))
      endif
      endif
      planeangle = epsf




c      tmp(1) = tgj*dcos(phi)*kp123(1)+tgj*dsin(phi)*kp123(2)
c     $ -dsqrt(1.-tgj**2)*kp123(3)
c      tmp(2) = -dsin(phi)*kp123(1) + dcos(phi)*kp123(2)
c      tmp(3) = dsqrt(1.-tgj**2)*dcos(phi)*kp123(1)
c     $ + dsqrt(1.-tgj**2)*dsin(phi)*kp123(2) + tgj*kp123(3)
c      tmp(4) = kp123(4)

c      tmp1(1) = tgj*dcos(phi)*km123(1)+tgj*dsin(phi)*km123(2)
c     $ -dsqrt(1.-tgj**2)*km123(3)
c      tmp1(2) = -dsin(phi)*km123(1) + dcos(phi)*km123(2)
c      tmp1(3) = dsqrt(1.-tgj**2)*dcos(phi)*km123(1)
c     $ + dsqrt(1.-tgj**2)*dsin(phi)*km123(2) + tgj*km123(3)
c      tmp1(4) = km123(4)

c      tmp2(1) = tgj*dcos(phi)*k0123(1)+tgj*dsin(phi)*k0123(2)
c     $ -dsqrt(1.-tgj**2)*k0123(3)
c      tmp2(2) = -dsin(phi)*k0123(1) + dcos(phi)*k0123(2)
c      tmp2(3) = dsqrt(1.-tgj**2)*dcos(phi)*k0123(1)
c     $ + dsqrt(1.-tgj**2)*dsin(phi)*k0123(2) + tgj*k0123(3)
c      tmp2(4) = k0123(4)


c      PP(1) = 0.d0
c      PP(2) = 0.d0
c      PP(3) = dsqrt((xm123**2-(xm12-xmp)**2)*(xm123**2-(xm12+xmp)**2)
c     $ /(4.*xm123**2))
c      PP(4) = dsqrt(xm12**2+PP(1)**2+PP(2)**2+PP(3)**2)
c      call boost(tmp,kp,PP,-1.d0)
c      call boost(tmp1,km,PP,-1.d0)


      PP(1) = -k0123(1)
      PP(2) = -k0123(2)
      PP(3) = -k0123(3)
      PP(4) = dsqrt(xm12**2+PP(1)**2+PP(2)**2+PP(3)**2)


      call boost(kp123,kp,PP,-1.d0)
      call boost(km123,km,PP,-1.d0)


c      write(*,*) km(1),km(2),km(3),
c     $ dsqrt(km(4)**2-km(1)**2-km(2)**2-km(3)**2)/xmp


      tgj_cm = kp(3)/dsqrt(kp(1)**2+kp(2)**2+kp(3)**2)
      co2a = kp(1)
     $ /(dsqrt(1.-tgj_cm**2)*dsqrt(kp(1)**2+kp(2)**2+kp(3)**2))
      si2a = kp(2)
     $ /(dsqrt(1.-tgj_cm**2)*dsqrt(kp(1)**2+kp(2)**2+kp(3)**2))
      if ((co2a.eq.0.).and.(si2a.eq.0.)) then
      epsf = 0.d0
      else
      if (si2a.ge.0) then
      epsf = dacos(co2a/dsqrt(co2a**2+si2a**2))
      else
      epsf = pi+dacos(-co2a/dsqrt(co2a**2+si2a**2))
      endif
      endif
      phi_cm = epsf


      arg(1) = xm12 
      arg(2) = xm123 
      arg(3) = tgj_cm 
      arg(4) = phi_cm 
      arg(5) = tgj_0 
      arg(6) = phi_0 
      arg(7) = tgj_t 
      arg(8) = phi_t 




      return 
      end



c---------------------- library -----------------------

      subroutine project(q,p,ex,ey,ez)
      implicit double precision (a-h,o-z)
      dimension q(4),p(4)
      dimension ex(3),ey(3),ez(3)
      p(1) = ex(1)*q(1)+ex(2)*q(2)+ex(3)*q(3)
      p(2) = ey(1)*q(1)+ey(2)*q(2)+ey(3)*q(3)
      p(3) = ez(1)*q(1)+ez(2)*q(2)+ez(3)*q(3)
      p(4) = q(4)
      return
      end 


      function sinv(p1,p2)
      implicit double precision (a-h,o-z)
      dimension p1(4),p2(4)
      sinv = (p1(4)+p2(4))**2 - (p1(1)+p2(1))**2 
     $         - (p1(2)+p2(2))**2 - (p1(3)+p2(3))**2 
      return
      end

      subroutine add(p1,p2,p)
      implicit double precision (a-h,o-z)
      dimension p1(4),p2(4),p(4)
      do i=1,4
      p(i) = p1(i) + p2(i)
      enddo
      return
      end 


      subroutine boost(q,p,PP,dir)
      implicit double precision (a-h,o-z)
      double precision M
      dimension q(4),p(4),PP(4)
      scal = q(1)*PP(1)+q(2)*PP(2)+q(3)*PP(3)
      M = dsqrt(PP(4)**2-PP(1)**2-PP(2)**2-PP(3)**2)
      do i=1,3
      p(i) = q(i) + scal*PP(i)/(M*(PP(4)+M)) + dir*q(4)*PP(i)/M
      enddo
      p(4) = q(4)*PP(4)/M + dir*scal/M 
      return
      end 


       FUNCTION ran2(idum)
       INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
       REAL*8 ran2,AM,EPS,RNMX
       PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     *   IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,
     *   IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2d-7,RNMX=1.d0-EPS)
       INTEGER idum2,j,k,iv(NTAB),iy
       SAVE iv,iy,idum2
       DATA idum2/123456789/, iv/NTAB*0/, iy/0/
       if (idum.le.0) then
           idum=max(-idum,1)
           idum2=idum
           do 11 j=NTAB+8,1,-1
               k=idum/IQ1
               idum=IA1*(idum-k*IQ1)-k*IR1
               if (idum.lt.0) idum=idum+IM1
               if (j.le.NTAB) iv(j)=idum
 11        enddo
           iy=iv(1)
       endif
       k=idum/IQ1
       idum=IA1*(idum-k*IQ1)-k*IR1
       if (idum.lt.0) idum=idum+IM1
       k=idum2/IQ2
       idum2=IA2*(idum2-k*IQ2)-k*IR2
       if (idum2.lt.0) idum2=idum2+IM2
       j=1+iy/NDIV
       iy=iv(j)-idum2
       iv(j)=idum
       if(iy.lt.1) iy=iy+IMM1
       ran2=min(AM*iy,RNMX)
       return
       END
c....................................................................


ccc---- biinibg 
      subroutine initbins(eg)
      implicit double precision (a-h,o-z)
      common /bins/ tgj_b_min(100),tgj_b_max(100),phi_b_min(100),
     $phi_b_max(100),xm123_b_min(100),xm123_b_max(100),s12_b_min(100),
     $s12_b_max(100),xm_min(100),xm_max(100)

      common /binfill/ n123(100),npm(100),n0p(100),nm0(100),ntgj_t(100),
     $nphi_t(100),ntgj_cm(100),nphi_cm(100),ntgj_0(100),nphi_0(100),
     $ np1x(100),np1y(100),np1z(100),np2x(100),np2y(100),np2z(100)
      common /bindal/ nd(100,100)
      common /bin2/ nbin

      pi = 4.d0*datan(1.d0)  
      xmp = 0.1395
      xmn = 0.9383
      s = 2.*eg*xmn + xmn**2    
      tgjmin = -1.d0
      tgjmax =  1.d0
      phimin = 0.d0
      phimax = 2.*pi       
      xm123min = 3.*xmp
      xm123max = dsqrt(s) - xmn
      s12min = 4.*xmp**2
c      s12max = (dsqrt(s) - xmn - xmp)**2
      s12max = (1.8 - xmp)**2
c      s12max = (2.-xmp)**2
c      s12max = (1.55 - xmp)**2

      xmommin = -1.
      xmommax = 1. 
      nbin = 100
      do i=1,nbin
      tgj_b_min(i) = dble(i-1)*(tgjmax-tgjmin)/nbin + tgjmin
      tgj_b_max(i) = dble(i)  *(tgjmax-tgjmin)/nbin + tgjmin
      phi_b_min(i) = dble(i-1)*(phimax-phimin)/nbin + phimin
      phi_b_max(i) = dble(i)  *(phimax-phimin)/nbin + phimin
      xm123_b_min(i) = dble(i-1)*(xm123max-xm123min)/nbin + xm123min
      xm123_b_max(i) = dble(i)  *(xm123max-xm123min)/nbin + xm123min
      s12_b_min(i) = dble(i-1)*(s12max-s12min)/nbin + s12min
      s12_b_max(i) = dble(i)*(s12max-s12min)/nbin + s12min
      xm_min(i) = dble(i-1)*(xmommax-xmommin)/nbin + xmommin
      xm_max(i) = dble(i  )*(xmommax-xmommin)/nbin + xmommin
      n123(i) = 0
      npm(i) = 0
      n0p(i) = 0
      nm0(i) = 0
      ntgj_t(i) = 0
      nphi_t(i) = 0
      ntgj_cm(i) = 0
      nphi_cm(i) = 0 
      ntgj_0(i) = 0
      nphi_0(i) = 0
      np1x(i) = 0
      np1y(i) = 0
      np1z(i) = 0
      np2x(i) = 0
      np2y(i) = 0
      np2z(i) = 0
      do j=1,nbin
      nd(i,j)  = 0. 
      enddo 
      enddo
      return
      end 


      
c--------------
      double complex function Y(l,m,x,p)
      implicit double precision (a-h,o-z)
      common /factor/ dgam(100)
      double complex xr,xi
      xr = (1.d0,0.d0)
      xi = (0.d0,1.d0)
      pi = 4.d0*datan(1.d0)

      dgam(1) = 1.d0
      do k=2,100
      dgam(k) = dgam(k-1)*dble(k-1)
      enddo

      goto 500

      if (l.eq.0) then
      Y = 1./dsqrt(4*pi)
      endif
      if (l.eq.1) then
      coe = dsqrt(3./(4.*pi))
      if (m.eq.0) then
      Y = coe*x*xr
      endif
      if (m.eq.1) then
      Y = -coe*dsqrt(1.-x**2)*(dcos(p)*xr + dsin(p)*xi)/dsqrt(2.d0)
      endif
      if (m.eq.-1) then
      Y = coe*dsqrt(1.-x**2)*(dcos(p)*xr - dsin(p)*xi)/dsqrt(2.d0)
      endif
      endif
      if (l.eq.2) then
      if (m.eq.0) then
      Y = dsqrt(5./(4.*pi))*((3./2.)*x**2-.5)*xr
      endif
      if (m.eq.1) then
      Y = -dsqrt(15./(8.*pi))*dsqrt(1.-x**2)*x*(dcos(p)*xr+dsin(p)*xi)
      endif
      if (m.eq.2) then
      Y = .25*dsqrt(15./(2.*pi))*(1.-x**2)*(dcos(2.*p)*xr+
     $ dsin(2.*p)*xi)
      endif
      if (m.eq.-1) then
      Y = dsqrt(15./(8.*pi))*dsqrt(1.-x**2)*x*(dcos(p)*xr-dsin(p)*xi)
      endif
      if (m.eq.-2) then
      Y = .25*dsqrt(15./(2.*pi))*(1.-x**2)*(dcos(2.*p)*xr-
     $ dsin(2.*p)*xi)
      endif
      endif
      if (l.eq.3) then
      if (m.eq.0) then
      Y = dsqrt(7./(16.*pi))*(5.*x**3-3.*x)*xr
      endif
      if (m.eq.1) then
      Y = -dsqrt(21./(64.*pi))*dsqrt(1.-x**2)*(5.*x**2-1.)
     $                *(dcos(p)*xr+dsin(p)*xi)
      endif
      if (m.eq.2) then
      Y = dsqrt(105./(32.*pi))*(1.-x**2)*x
     $ *(dcos(2.*p)*xr+dsin(2.*p)*xi)
      endif
      if (m.eq.3) then
      Y = -dsqrt(35./(64.*pi))*(dsqrt(1.-x**2))**3
     $ *(dcos(3.*p)*xr+dsin(3.*p)*xi)
      endif
      if (m.eq.-1) then
      Y = dsqrt(21./(64.*pi))*dsqrt(1.-x**2)*(5.*x**2-1.)
     $                *(dcos(p)*xr-dsin(p)*xi)
      endif
      if (m.eq.-2) then
      Y = dsqrt(105./(32.*pi))*(1.-x**2)*x
     $ *(dcos(2.*p)*xr-dsin(2.*p)*xi)
      endif
      if (m.eq.-3) then
      Y = dsqrt(35./(64.*pi))*(dsqrt(1.-x**2))**3
     $ *(dcos(3.*p)*xr-dsin(3.*p)*xi)
      endif
      endif
      if (l.eq.4) then
      if (m.eq.0) then
      Y = dsqrt(9./(256.*pi))*(35.*x**4-30.*x**2+3.)*xr
      endif
      if (m.eq.1) then
      Y = -dsqrt(45./(64.*pi))*dsqrt(1.-x**2)*(7.*x**3-3.*x)
     $                *(dcos(p)*xr+dsin(p)*xi)
      endif
      if (m.eq.2) then
      Y = dsqrt(45./(128.*pi))*(1.-x**2)*(7.*x**2-1.)
     $ *(dcos(2.*p)*xr+dsin(2.*p)*xi)
      endif
      if (m.eq.-1) then
      Y = dsqrt(45./(64.*pi))*dsqrt(1.-x**2)*(7.*x**3-3.*x)
     $                *(dcos(p)*xr-dsin(p)*xi)
      endif
      if (m.eq.-2) then
      Y = dsqrt(45./(128.*pi))*(1.-x**2)*(7.*x**2-1.)
     $ *(dcos(2.*p)*xr-dsin(2.*p)*xi)
      endif
      if (m.eq.-3) then
      Y = 3.*dsqrt(35./(64.*pi))*(dsqrt(1.-x**2))**3*x
     $ *(dcos(3.*p)*xr-dsin(3.*p)*xi)
      endif
      if (m.eq.-4) then
      Y = 3.*dsqrt(35./(2.*pi))*(dsqrt(1.-x**2))**4
     $ *(dcos(4.*p)*xr-dsin(4.*p)*xi)/16.
      endif
      endif
      goto 400


500   continue
      sum = 0.d0
      do is=0,l
      if ( ( (l-m-is).ge.0).and.( (m+is).ge.0) ) then
      sum = sum
     $      +
     $ (dgam(l+1)/(dgam(l-m-is+1)*dgam(m+is+1)))
     $     *(dgam(l+1)/(dgam(l-is+1)*dgam(is+1)))
     $     *(-1)**(l-is)
     $     *dsqrt( (1. + x)/2. )**(2*is + m)
     $     *dsqrt( (1. - x)/2. )**(2*l - 2*is - m)
       endif
       enddo
      sum = sum*dsqrt( (2.*dble(l) + 1.)/(4.*pi) )
     $       * dsqrt( dgam(l+m+1)*dgam(l-m+1) )/dgam(l+1)
       Y = sum*(dcos(m*p)*xr + dsin(m*p)*xi)
400   continue
      return
      end



      subroutine frac
      implicit double precision (a-h,o-z)
      common /factor/ dgam(100)
      pi = 4.d0*datan(1.d0)
      dgam(1) = 1.d0
      do k=2,100
      dgam(k) = dgam(k-1)*dble(k-1)
      enddo
      return
      end
c-------------------------------------------

      function fac(n)
      implicit double precision (a-h,o-z)

      if (n.eq.0) then
      prod = 1.
      else

      prod = 1.
      do i=1,n
      prod = prod*dble(i)
      enddo

      endif

      fac = prod

      return
      end





      function cg(j1,m1,j2,m2,j,m)
      implicit double precision (a-h,o-z)

      if ((iabs(m1).gt.j1).or.(iabs(m2).gt.j2).or.(iabs(m).gt.j)) then
      cg = 0.
      return
      endif

      if ((j.gt.(j1+j2)).or.(j.lt.iabs(j1-j2))) then
      cg = 0.
      return
      endif

      if ((m1+m2).ne.m) then
      cg = 0.
      return
      endif

      if ((j1.eq.0).or.(j2.eq.0)) then
      cg = 1.
      return
      endif


      sum  = 0.


c      icount = 0

      do i=0,100

      if( dble(i/2).eq.dble(i)/2 )then
      coef = 1.
      else
      coef = -1.
      endif



c      write(*,*) i,coef

c      if (i.eq.0) then
c      coef = 1.
c      else
c      coef = (-1.)**i
c      endif


      if (((j1+j2-j-i).lt.0).or.((j1-m1-i).lt.0).or.((j2+m2-i).lt.0)
     $   .or.((j-j2+m1+i).lt.0).or.((j-j1-m2+i).lt.0)) goto 999

c       icount = icount + 1

      
       ala = 
     $   coef/(fac(i)*fac(j1+j2-j-i)*fac(j1-m1-i)*fac(j2+m2-i)
     $         *fac(j-j2+m1+i)*fac(j-j1-m2+i))
       sum = sum + ala

c       write(*,*) i,icount,ala
 999   continue

       enddo

       cg = sum*dsqrt((2*j+1)*fac(j1+j2-j)*fac(j1-j2+j)
     $   *fac(-j1+j2+j)/fac(j1+j2+j+1))
     $  *dsqrt(fac(j1+m1)*fac(j1-m1)*fac(j2+m2)*fac(j2-m2)*fac(j+m)
     $   *fac(j-m))


       
       return
       end



 		subroutine fdai(fun,a,b,c,d,max,dokl,wyn,t)
C       * * * * * * * * * * * * * * * * * * * * * * * *
C       * PROCEDURA CALKUJACA FUNKCJE 2-ch ZMIENNYCH  *
C       * * * * * * * * * * * * * * * * * * * * * * * *
		implicit double precision(a-h,o-z)
		logical t,t1
		external fun
		common /dupa/ d1,d2,e1,e2
		dimension ist1(36),ist2(36),err(8),num(8)
		idod=4
  2		l=0
		wyn=1.d-35
		wyn1=1.d-35
		m=1
		ist1(1)=0
		ist2(1)=0
		t1=t
		t=.true.
		dz1=0.5d0*(b-a)
		dz2=0.5d0*(d-c)
		l1=0
		l2=0
  5		d1=dz1/2.d0
		d2=dz2/2.d0
		a1=a+dble(2*l1)*dz1+d1
		c1=c+dble(2*l2)*dz2+d2
  6		do 20 i=1,idod
		k=i-1
		e1=a1+dble(mod(k,2))*dz1
		e2=c1+dble(mod(k/2,2))*dz2
		call gaus46(fun,res,res1)
 18		eps=dabs((res-res1)/(wyn+res))
		if(eps.le.dokl.or.m.eq.max) go to 15
		l=l+1
		num(l)=k+idod
		err(l)=eps
		k=l
 14		if(k.ne.1) go to 60
		go to 20
 60		continue
		if(err(k).le.err(k-1)) go to 20
		pom=err(k)
		err(k)=err(k-1)
		err(k-1)=pom
		ipo=num(k)
		num(k)=num(k-1)
		num(k-1)=ipo
		k=k-1
		go to 14
 15		wyn=wyn+res
		wyn1=wyn1+res1
		t=(t.and.((m.ne.max).or.(eps.le.dokl)))
 20		continue
		if(l.eq.0) go to 31
		do 27 i=1,l
		ist1(m)=4*ist1(m)+2+mod(num(i),2)
		ist2(m)=4*ist2(m)+2+mod(num(i)/2,2)
 27		continue
		m=m+1
		dz1=0.5d0*dz1
		dz2=0.5d0*dz2
		l1=2*l1+mod(ist1(m-1),2)
		l2=2*l2+mod(ist2(m-1),2)
		l=0
		ist1(m)=0
		ist2(m)=0
		go to 5
 31		if(m.eq.1) go to 40
                ist1(m-1)=ist1(m-1)/4
		ist2(m-1)=ist2(m-1)/4
		if(ist1(m-1).ne.0) go to 50
		m=m-1
		dz1=2.d0*dz1
		dz2=2.d0*dz2
		l1=l1/2
		l2=l2/2
		go to 31
 50             continue
                l1=(l1/2)*2+mod(ist1(m-1),2)
		l2=(l2/2)*2+mod(ist2(m-1),2)
		go to 5
 40		dokl=dabs((wyn-wyn1)/wyn)
		return
		end
		subroutine gaus46(fun,resg,resb)
		implicit double precision(a-h,o-z)
		common /dupa/ d1,d2,e1,e2
	        external fun
        	dimension g(2,3),g4(2,2),g6(2,3)
		data g4 /.86113631159d0,.34785484514d0,
     1		.33998104358d0,.65214515486d0/
		data g6 /.93246951420d0,.17132449283d0,
     1               .66120938647d0,.36076157305d0,
     2		.23861918408d0,.46791393457d0/
		d=d1*d2
		lim=2
		do 15 i=1,2
		do 15 j=1,2
 15		g(i,j)=g4(i,j)
  1		gaus=0.d0
  4		do 10 j=1,lim
		y=d2*g(1,j)
		znak=1.d0
 11		s=e2+znak*y
		znak=-znak
		do 6 n=1,lim
		x=d1*g(1,n)
		c1=fun(e1+x,s)+fun(e1-x,s)
 6		gaus=gaus+c1*g(2,n)*g(2,j)
		if(znak.eq.-1.d0) go to 11
 10		continue
  7		if(lim.eq.3) go to 9
		lim=3
		do 8 i=1,2
		do 8 j=1,3
  8		g(i,j)=g6(i,j)
		resb=gaus*d
		go to 1
  9		resg=gaus*d
		return
		end



c----------------------------------------------------------
   	subroutine intx(f,a,b,n,eps,result,t)
	implicit double precision (a-h,o-z)
	integer *4 iadr(1600)
	dimension aa(1600),bb(1600)
c       real eps,eps1,eps2
	logical t
	external f
	t=.true.
	result=0.d0
	reslt2=0.d0
	ind=1
	iadr(1)=-1
	aa(1)=a
	bb(1)=b
   1  c=(aa(ind)+bb(ind))/2.
            call gauscx(f,aa(ind),c,ra1,ra2)
            eps1=dabs(ra1-ra2)/(dabs(ra1+result)+1.0d-30)
            call gauscx(f,c,bb(ind),rb1,rb2)
            eps2=dabs(rb1-rb2)/(dabs(rb1+result)+1.0d-30)
  	if(eps1-eps2) 10,10,20
  10  if(eps1-eps) 12,12,11
  11  if(ind-n) 13,15,15
  15  t=.false.
  12  result=result+ra1
      reslt2=reslt2+ra2
      iadr(ind)=iadr(ind)+100
      if(iadr(ind)-150) 20,20,30
  13  ind=ind+1
      iadr(ind)=0
      aa(ind)=aa(ind-1)
      bb(ind)=(aa(ind-1)+bb(ind-1))/2.
      go to 1
  14  iadr(ind)=iadr(ind)+100
      if(iadr(ind)-150) 23,23,30
  20  if(eps2-eps) 22,22,21
  21  if(ind-n) 23,25,25
  25  t=.false.
  22  result=result+rb1
      reslt2=reslt2+rb2
      iadr(ind)=iadr(ind)+100
      if(iadr(ind)-150) 10,10,30
  23  ind=ind+1
      iadr(ind)=1
      aa(ind)=(aa(ind-1)+bb(ind-1))/2.
      bb(ind)=bb(ind-1)
      go to 1
  24  iadr(ind)=iadr(ind)+100
      if(iadr(ind)-150) 13,13,30
  30  ind=ind-1
      if(iadr(ind+1)-200) 100,14,24
 100  eps=dabs(result-reslt2)/(dabs(result)+1.d-30)
      return
      end
      subroutine gauscx(f,a,b,gauskr,gaus)
      implicit double precision (a-h,o-z)
      dimension g(3,8)
      external f
      data g/
     $9.933798 7588 1716d-1,0.                   ,1.782238 3320 7104d-2,
     $9.602898 5649 7536d-1,1.012285 3629 0376d-1,4.943939 5002 1394d-2,
     $8.941209 0684 7456d-1,0.                   ,8.248229 8931 3584d-2,
     $7.966664 7741 3626d-1,2.223810 3445 3374d-1,1.116463 7082 6840d-1,
     $6.723540 7094 5158d-1,0.                   ,1.362631 0925 5172d-1,
     $5.255324 0991 6329d-1,3.137066 4587 7887d-1,1.566526 0616 8188d-1,
     $3.607010 9792 8132d-1,0.                   ,1.720706 0855 5211d-1,
     $1.834346 4249 5650d-1,3.626837 8337 8362d-1,1.814000 2506 8035d-1/
      data g39/1.844464 0574 4692d-1/
      gaus=0.d0
      gauskr=0.d0
      d=(b-a)/2.
      e=(b+a)/2.
      do 100 l=1,8
      x=d*g(1,l)
      c=f(e+x)+f(e-x)
      gaus=gaus+c*g(2,l)
      gauskr=gauskr+c*g(3,l)
 100  continue
      gaus=d*gaus
      gauskr=d*(gauskr+g39*f(e))
      return
      end



