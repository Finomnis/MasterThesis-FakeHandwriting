cmake_minimum_required(VERSION 3.9)
project(MasterThesis NONE)

set(MAIN_TEX mt)

# Get all tex files in input directory
file(GLOB TEX_FILES RELATIVE ${CMAKE_SOURCE_DIR} "*.tex")

# All outputs of pdflatex
set(PDFLATEX_OUTPUTS
  ${MAIN_TEX}.bcf
  ${MAIN_TEX}.glo
  ${MAIN_TEX}.ist
  ${MAIN_TEX}.lof
  ${MAIN_TEX}.log
  ${MAIN_TEX}.lot
  ${MAIN_TEX}.out
  ${MAIN_TEX}.pdf
  ${MAIN_TEX}.run.xml
  ${MAIN_TEX}.synctex.gz
  ${MAIN_TEX}.toc
)
foreach(tex_file ${TEX_FILES})
  get_filename_component(tex_file_aux ${tex_file} NAME_WE)
  list(APPEND PDFLATEX_OUTPUTS ${tex_file_aux}.aux)
endforeach()

# All outputs of makeglossaries
set(GLOSS_OUTPUTS
  ${MAIN_TEX}.glg
  ${MAIN_TEX}.gls
)

# All outputs of biber
set(BIBER_OUTPUTS
  ${MAIN_TEX}.bbl
  ${MAIN_TEX}.blg
)

# The pdflatex command, to avoid repetitions
set (PDFLATEX_COMMAND COMMAND pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error -output-directory ${CMAKE_CURRENT_BINARY_DIR} ${MAIN_TEX}.tex BYPRODUCTS ${PDFLATEX_OUTPUTS} DEPENDS ${TEX_FILES} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )


### The actual steps ######
add_custom_command(OUTPUT pass.gloss COMMENT "MakeGlossaries" DEPENDS pass.1 BYPRODUCTS ${GLOSS_OUTPUTS} COMMAND makeglossaries -q -d ${CMAKE_CURRENT_BINARY_DIR} ${MAIN_TEX} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/pass.gloss)

add_custom_command(OUTPUT pass.biber COMMENT "Biber" DEPENDS pass.gloss ${MAIN_TEX}.bib BYPRODUCTS ${BIBER_OUTPUTS} COMMAND biber -q --output-directory ${CMAKE_CURRENT_BINARY_DIR} ${MAIN_TEX} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/pass.biber)

add_custom_command(OUTPUT pass.1 COMMENT "PdfLatex Pass 1" ${PDFLATEX_COMMAND} COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/pass.1)
add_custom_command(OUTPUT pass.2 DEPENDS pass.1 pass.biber COMMENT "PdfLatex Pass 2" ${PDFLATEX_COMMAND} COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/pass.2)
add_custom_command(OUTPUT pass.3 DEPENDS pass.2 COMMENT "PdfLatex Pass 3" ${PDFLATEX_COMMAND} COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/pass.3)

# Final pseudo-command, to attach the chain to ALL
add_custom_target(pdf ALL DEPENDS pass.3)
