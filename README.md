# Automated Tool for Modeling Impedance for Spice (ATMIS)

## Description
The ATMIS is a tool designed to automatedly generate an LTSpice model, given the frequency
dependent impedance data of a measurement.

Data needs to be provided to the software in Touchstone (.sNp) file format. The tool will
then calculate the impedance data of the device under test (DUT) and proceed with a series
of least squares optimization routines to build a suitable *LTSpice* model.

The models obtained consist of serially chained parallel resonance circuits for inductors or
parallel series resonance circuits for capacitors.

The tool is specialized in fitting passive components with DC bias and generating *LTSpice* 
models that are current or voltage dependent.

## System Requirements
- Python version 3.10 or newer.
- Around 2 Gb of available Memory
- A good CPU (any CPU will work fine, but the calculation time can become quite high)

If you find yourself running into memory issues, you can adjust the *MULTIPROCESSING_COUNT*
variable in the config.py file. This determines how many processes will be started for
fitting; less processes will go easier on RAM usage at the cost of time.

## Usage

### Start-up
Run ATMIS via the `main.py` file.

When the tool is run, the user will see a GUI, consisting of the *Specification* area, a *file
list* that will be filled and is empty at first execution and a *log window* which is also empty
upon first execution.

To fit a component, one must first select the type of component in the dropdown menu in the left
upper corner (**NOTE**: currently only capacitors and inductors are supported). Furthermore, the
calculation mode for impedance has to be selected in the mid-right area spelling
"Z Calculation Method". Select "Series Through" or "Shunt Through" depending on how the DUT
was attached to the measurement setup.

### Loading Files
After that click on the "Select s2p File(s)" button to load the files for the DUT. The files
will be loaded and displayed in the *file list pane* on the right, together with a radiobutton and
a small entry box.

### Getting Things Ready
To tell the tool the DC Bias for each file, write the DC Bias values into the entry boxes right
of the filename in the *file list pane*. The units for the DC Bias are Amperes or Volts for
inductors and capacitors respectively (**NOTE**: engineering or scientific notation is not yet
supported).

Also, a reference file needs to be selected, this can be done by clicking the radiobutton
left to the file which is the reference file. The reference file is the one, which **does not**
have any DC Bias.

The nominal value of the component can be provided in the *specification area* on the top
left of the GUI in Henry or Farads for inductors or capacitors respectively. However, the tool
has a built-in calculation for the nominal inductance/capacitance which works quite well.

The series resistance of the component can also be input at the *specification area*. The tool
has a calculation method for that as well though.
**NOTE**: It is highly recommended to provide the series resistance for any inductor you might
want to fit, since **the series resistance calculation does not work well for inductors**. It
does, however, work fairly well for capacitors.

The prominence for the fit can also be adjusted. This parameter needs to be input in dB and
determines the prominence a peak has to have in order to be fit as a resonance circuit.

# Further notes

## Config File

Certain parameters can be adjusted in the config.py file:
- FREQ_UPPER_LIMIT: The highest frequency to which the fitter will handle the data.
- CAPTYPE: if you want to fit the acoustic resonance of an MLCC, set this constant
    to *constants.MLCC*, otherwise set to *constants.GENERIC*.
- FULL_FIT: if you want the model to be fully parametric, i.e. all resonance circuits
    to be DC bias dependent (as opposed to only the main resonance parameters), set to **1**.
- MULTIPROCESSING_COUNT: determines how many processes will be started for the fit.

## Known issues
- The GUI does not have responsive design, so depending on your screen resolution, things might
look a bit weird.
- Some coils/capacitors might not be detected by the tool, that is because the
inductive/capacitive range is determined by the phase of the dataset. If you
desperately need to fit a part that runs into this issue, you can go to the
*constants.py* file and set the *PERMITTED_MIN_PHASE* to a lower value. However,
do not expect perfect results from that workaround.





