%
% Read datafile from shaker experiment
%
clear all
PSDataFile=['/Users/syedather/Downloads/PS2DataFile_Sept2016']; % This file contains three vectors:

% TimeExp: series of 500000 time stamps increasing at the 10ms sample rate
% DataExpLowTemp: 500000 measurements of impact sensor data for experiment
%                   with shaker setting low (mimicking a relatively low
%                   temperature)
% DataExpHighTemp: 500000 measurements at high shaker setting (i.e.
%                   relatively high temperature)
%
load(PSDataFile); % read the datafile
%
%%
%
% Fontsize definitions (just to get nice text size on plots)
%
FST=14; % Fontsize for titles
FSL=12; % Fontsize for axis labels
%
TPlotBegin=0; TPlotEnd=10; % time window for plots
VPlotMin=-100; VPlotMax=500; % Velocity scales for plotting
%
TimeBin=TimeExp(2)-TimeExp(1);
% Parameters for detecting peaks, used by the "findpeaks" command (see
% below)
%
ThrVel=50; % Velocity threshold values (mm/s), used to avoid noise
% fluctuations
HalfPeakWidth=3; % halfwidth for analyzing peak shape, to avoid detecting
% glitches
MinPeakDist=round(0.25/TimeBin); % a minimum distance between peaks is
% enforced in order to prevent problems when peaks cluster too juch
% together. This means that not all peaks will be detected, and the
% shortest intervals will be ignored, as you can see from the interval
% distribution plotted below
%

for ExpTemp=1:2
    switch ExpTemp
        case 1
            DataExp=DataExpLowTemp;
            ExpLabel='Low Temp'
            FigureBase=0;
        case 2
            DataExp=DataExpHighTemp;
            ExpLabel='High Temp'
            FigureBase=10;
    end
    %
    %%
    % Analyze the data trace to determine peak positions and amplitudes;
    % read the comments below; in practive not all peaks can be detected (due
    % to noise, electronic glitches, and collisions occurring too close
    % together in time
    %
    [PeakAmpl,PeakInd]=findpeaks(DataExp,'MinPeakHeight',ThrVel,...
        'MinPeakWidth',HalfPeakWidth+1,'MinPeakDistance',MinPeakDist);
    PeakAmpl=PeakAmpl(1:end-1);
    PeakInd=PeakInd(1:end-1);
    NPeaks=length(PeakInd);
    PeakTimes=TimeExp(PeakInd); % list of times of positive peak occurrences
    %
    figure(FigureBase+1);clf
    subplot(2,1,1)
    %
    % plot the recorded trace of the impact sensor and mark the peak
    % rising and falling edge, and location of peak maxima
    %
    PeakIndx1=PeakInd-HalfPeakWidth; % indices of left edge
    PeakIndx2=PeakInd+HalfPeakWidth; % indices of right edge
    PeakPos=[PeakInd];
    %
    plot(TimeExp,DataExp,'-k',...
        TimeExp(PeakIndx1),DataExp(PeakIndx1),'or',...
        TimeExp(PeakIndx2),DataExp(PeakIndx2),'ob',...
        TimeExp(PeakPos),DataExp(PeakPos),'*g')
    axis([TPlotBegin,TPlotEnd,VPlotMin,VPlotMax])
    title(['Sensor trace with peaks detected, ' ExpLabel],'FontSize',FST)
    xlabel('time (s)','FontSize',FSL)
    ylabel('sensor velocity (mm/s)','FontSize',FSL)
    axis([TPlotBegin,TPlotEnd,VPlotMin,VPlotMax]);
    %
    subplot(2,1,2)
        stem(TimeExp(PeakPos),DataExp(PeakPos),'k')
    axis([TPlotBegin,TPlotEnd,VPlotMin,VPlotMax])
    title(['Velocity Peak Events, ' ExpLabel],'FontSize',FST)
    xlabel('time (s)','FontSize',FSL)
    ylabel('sensor velocity (mm/s)','FontSize',FSL)
    axis([TPlotBegin,TPlotEnd,VPlotMin,VPlotMax]);
    %
    %%
    % Statistics of time intervals between peaks
    %
    % Top: count histogram in linear-linear scales
    % Bottom: Normalized Interval Distribution of a negative 
    % exponential with vertical log scale; an exponential function 
    % should form a straight line in this plot. 
    %    
    figure(FigureBase+2);clf
    TMax=5;
    TBin=TMax/100; % binsize for histogram
    THist=[0:TBin:TMax]; % set of time bins for the histogram
    %
    TimeIntervals=diff(PeakTimes); % time intervals are the difference  of successive impact times
    TotCount=length(TimeIntervals); % the total number of time intervals counted
    IntCount=hist(TimeIntervals,THist); % count in these bins
    IntProbDens=IntCount/(sum(IntCount)*TBin); % Prob. DENSITY for intervals has units of 1/s, and integrates to 1
    subplot(2,3,1)
    bar(THist,IntCount) % plot histogram as counts in a bar graph
    xlabel('\Deltat (s)')
    ylabel('Count')
    axis([0,TMax,0,inf])
    title([{[ExpLabel]},{'Counts of \Deltat'},{['bin=' num2str(TBin) ' s, N=' num2str(TotCount)]}],'FontSize',FST)
    subplot(2,3,4)
    semilogy(THist(1:end-1),IntProbDens(1:end-1),'k-o','LineWidth',2) % plot histogram on logarithmic vertical axis
    xlabel('\Deltat (s)')
    ylabel('P(\Deltat) (1/s)')
    axis([0,TMax,max(IntProbDens)/1000,max(IntProbDens)])
    title('Interval Distribution','FontSize',FST)
    %
    %%
    %
    % Statistics of peak velocities
    %
    % Top: Count histogram in linear-linear scales
    % Bottom: Probability density in linear-log scales
    % On this plot, a Gaussian shoule form an inverted parabola with its
    % maximum at V=0. Note, the plot shows only positive values for V, 
    % because the sensor is unidirectional
    %
    PeakVelMax=500;
    PeakVelBin=PeakVelMax/100; % binsize for amplitude histogram
    PeakVelVals=[0:PeakVelBin:PeakVelMax]; % set up amplitude bins for the histogram
    PeakVelCount=hist(DataExp(PeakPos),PeakVelVals); % count in these bins
    TotPeakVelCount=length(PeakVelCount);
    PeakVelProbDens=PeakVelCount/(TotPeakVelCount*PeakVelBin); % Prob DENSITY for peak values
    subplot(2,3,2)
    bar(PeakVelVals,PeakVelCount) % plot histogram as a bar graph
    xlabel('Peak velocity (mm/s)')
    ylabel('Count')
    axis([0,PeakVelMax,0,inf])
    title([{'Counts of V'},{['bin=' num2str(PeakVelBin) ' mm/s']}],'FontSize',FST)
    subplot(2,3,5)
    semilogy(PeakVelVals(1:end-1),PeakVelProbDens(1:end-1),'k-o','LineWidth',2) % plot histogram on logarithmic vertical axis
    title('Velocity Prob Dist','FontSize',FST)
    xlabel('V (mm/s)')
    ylabel('P(V) (s/mm)')
    axis([0,PeakVelMax,max(PeakVelProbDens)/100,max(PeakVelProbDens)])
    %
    %%
    %
    % Statistics of (velocity)^2
    %
    % Top: histogram of values of (velocity)^2; this should be 
    % proportional to kinetic energy
    % Bottom: Probability distribution and two fits
    %
    PeakVelSqMax=2.5*10^5;
    PeakVelSqBin=PeakVelSqMax/100; % binsize for 'energy' histogram
    PeakVelSqVals=[0:PeakVelSqBin:PeakVelSqMax]; % set up amplitude bins for the histogram
    PeakVelSqCount=hist((DataExp(PeakPos)).^2,PeakVelSqVals); % count in these bins
    TotPeakVelSqCount=length(PeakVelSqCount);
    PeakVelSqProbDens=PeakVelSqCount/(TotPeakVelSqCount*PeakVelSqBin); % Prob DENSITY for peak values
    subplot(2,3,3)
    bar(PeakVelSqVals,PeakVelSqCount) % plot histogram as a bar graph
    xlabel('V^2 (mm/s)^2')
    ylabel('Count')
    axis([0,PeakVelSqMax,0,inf])
    title([{'Counts of V^2'},{['bin=' num2str(PeakVelSqBin) ' mm/s^2']}],'FontSize',FST)
    %
    subplot(2,3,6)
    semilogy(PeakVelSqVals(1:end-1),PeakVelSqProbDens(1:end-1),'k-o','LineWidth',2) % plot histogram on logarithmic vertical axis
        xlabel('V^2 (mm/s)^2')
    ylabel('P(V^2) (s/mm)^2')
    axis([0,PeakVelSqMax,max(PeakVelSqProbDens)/1000,max(PeakVelSqProbDens)])
    title(['P(V^2)'],'FontSize',FST)
end

