# Windows PiPL resource, exactly as plugin\build.bat makes it (cmake -P script, run in the build directory):
#   cl /nologo [/DOB_KANTAN] /I <SDK>\Headers /EP OpticBokehPiPL.r > PiPL.rr
#   <SDK>\Resources\PiPLtool.exe PiPL.rr PiPL.rrc          (bare file names: PiPLtool cannot open non-ASCII paths)
#   cl /nologo /D MSWindows /EP PiPL.rrc > PiPL.rc         (rc then compiles PiPL.rc as a target source)
# Inputs: -DSDK=<Examples dir> -DR=<OpticBokehPiPL.r> -DDIR=<output dir, ASCII> -DKANTAN=0|1 -DCL=<cl.exe>
set(defs "")
if(KANTAN)
  set(defs /DOB_KANTAN)
endif()
execute_process(COMMAND ${CL} /nologo ${defs} /I ${SDK}/Headers /EP ${R}
  WORKING_DIRECTORY ${DIR} OUTPUT_FILE ${DIR}/PiPL.rr RESULT_VARIABLE r1)
if(NOT r1 EQUAL 0)
  message(FATAL_ERROR "cl /EP of the PiPL .r failed (${r1})")
endif()
file(REMOVE ${DIR}/PiPL.rrc)
execute_process(COMMAND ${SDK}/Resources/PiPLtool.exe PiPL.rr PiPL.rrc WORKING_DIRECTORY ${DIR} RESULT_VARIABLE r2)
if(NOT r2 EQUAL 0 OR NOT EXISTS ${DIR}/PiPL.rrc)
  message(FATAL_ERROR "PiPLtool failed (${r2})")
endif()
execute_process(COMMAND ${CL} /nologo /D MSWindows /EP PiPL.rrc
  WORKING_DIRECTORY ${DIR} OUTPUT_FILE ${DIR}/PiPL.rc RESULT_VARIABLE r3)
if(NOT r3 EQUAL 0)
  message(FATAL_ERROR "cl /EP of PiPL.rrc failed (${r3})")
endif()
