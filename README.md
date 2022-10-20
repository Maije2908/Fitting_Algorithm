# Automated Tool for Modeling Impedance for Spice (ATMIS)

## Description
The ATMIS is a tool designed to automatedly generate an LTSpice model, given the frequency
dependent impedance data of a measurement.

Data needs to be provided to the software in Touchstone (.sNp) file format. The tool will
then calculate the impedance data of the device under test (DUT) and proceed with a series
of least squares optimization routines to build a suitable *LTSpice* model.

The tool is specialized in fitting passive components with DC bias and generating *LTSpice* 
models that are current or voltage dependent.

## Usage

### Start-up
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

The prominence for the fit can also be adjusted.
TODO: hier noch irgendwie einbringen, was die prominence macht und vielleicht in der Description
noch angeben, wie genau wir die Modelle basteln, damit man versteht, worum es bei dem parameter
geht

# Further notes

TODO: hier anmerken, dass die MLCCs in der config selektiert werden müssen
TODO: hier anmerken, dass die Größe vom Multiprocessing pool angepasst werden kann
TODO: hier anmerken, dass die GUI kein responsive design hat
TODO: hier anmerken, wieviel RAM das ding ca braucht und dass man die multiprocessing size heruntersetzen kann
TODO: hier anmerken, dass die Software ein paar bugs hat
TODO: hier anmerken, dass die inductive/capacitive range nicht immer erkannt wird, je nach phase








