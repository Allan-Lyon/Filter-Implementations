%% Preliminaries
% This clears all variables and sets the format to disp more digits.
clearvars
close all
clc
format long

%% Addpath to Attitude Representations Folder
addpath('../Attitude Representations')

%% Addpath to Attitude Kinematics Folder
addpath('../Attitude Kinematics')

%% Addpath to Attitude Dynamics Folder
addpath('../Attitude Dynamics')

%% Load qBus.mat
load qBus.mat;

%% Load the Mass Properites
mass_properties;

%%
% Simulation time parameters
tsim = 60;
dt_sim = 0.001;
dt_gyros = 0.01;
dt_v1 = 1;
dt_v2 = 10;

sigmav1 = 1*pi/180;
sigmav2 = 0.1*pi/180;

sigmaw = 1*pi/180;
sigmab = 0.1*pi/180;

v1_I = [1;0;0];
v2_I = [0;1;0];

%% Initial Conditions
angle = 45*pi/180;
axis = [1;1;1]; axis=axis/norm(axis);
q0_BI = e2q(axis,angle);
A0_BI = q2A(q0_BI);
A0_IB = A0_BI';

wbi0_B = [-10;20;30]*pi/180;

v1_B = A0_BI*v1_I + sigmav1*randn(3,1);
v2_B = A0_BI*v2_I + sigmav2*randn(3,1);

Ahat0_BI = triad(v2_B, v2_I, v1_B, v1_I);
qhat0_BI = A2q(Ahat0_BI);

sigmaangle = (60*pi/180)^2;
sigmaomega = (30*pi/180)^2;
P0 = diag([sigmaangle, sigmaangle, sigmaangle, sigmaomega, sigmaomega, sigmaomega]);

% Seed values for noise parameters
nb_seed = randi([0, 2^32], 3, 1);
nw_seed = randi([0, 2^32], 3, 1);
nv1_seed = randi([0, 2^32], 3, 1);
nv2_seed = randi([0, 2^32], 3, 1);
% nb_seed = [1;2;3];
% nw_seed = [4;5;6];
% nv1_seed = [7;8;9];
% nv2_seed = [10;11;12];

Bw0 = [10;-20;-30]*pi/180;
Bwhat0 = [0;0;0];