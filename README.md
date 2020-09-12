# Diffusion Simulator

## Introduction
This program simulates the diffusion of oxygen from capillaries to the 
surrounding tissue, both on a repeating 2x2 unit and a 3x3 repeating unit. 
There is also the option to select 4 way or 8 way diffusion from the 
capillaries, as well as change the starting percentage of capillaries that 
are functional.

## Variables
This program contains the following editable variables:
- Region size
- Diffusion constant
- Capillary starting value
- Tissue starting value
- Capillary death value
- Tissue death value
- Distance step
- Time step
- Percent of functional capillaries (selected at random)
* The units of these variables (if applicable) will be provided via 
comment next to the line they occur on.

## Functionality
This program contains the following functionality:
- An interactive plot to visually track the progression of diffusion
- A graph of both capillary and tissue deaths over time
- A graph of a selectable capillary's saturation in percent over time
- A graph of the average tissue saturation in percent over time
- A adjustable number representing the percentage of capillaries disabled 
at random at the start of the simulation

## How to Run
- Install python3
- Install dependencies (numpy, scipy, matplotlib)
- Run SimRunner.py
- Instructions on changing variables will be in comments at the bottom of SimRunner

