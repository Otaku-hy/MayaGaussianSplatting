cmake_minimum_required(VERSION 3.20)

project(GaussianSplatting LANGUAGES CXX)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR
        "Maya GaussianSplatting only supports x64 platform now.")
endif()



if(NOT MAYA_LOCATION)
    set(MAYA_LOCATION "$ENV{MAYA_LOCATION}")
endif()
if(NOT DEVKIT_LOCATION)
    set(DEVKIT_LOCATION "$ENV{DEVKIT_LOCATION}")
endif()

if(NOT MAYA_LOCATION OR NOT EXISTS "${MAYA_LOCATION}")
    message(FATAL_ERROR
        "MAYA_LOCATION is not set or does not exist.\n"
        "Set it as an environment variable or pass -DMAYA_LOCATION=...\n")
endif()

if(NOT DEVKIT_LOCATION OR NOT EXISTS "${DEVKIT_LOCATION}")
    message(FATAL_ERROR
        "DEVKIT_LOCATION is not set or does not exist.\n"
        "Set it as an environment variable or pass -DDEVKIT_LOCATION=...\n")
endif()

message(STATUS "MAYA_LOCATION   = ${MAYA_LOCATION}")
message(STATUS "DEVKIT_LOCATION = ${DEVKIT_LOCATION}")

set(PLUGIN_TARGET_PATH "" CACHE PATH
    "Where to deploy the built .mll and shaders. Leave empty to use the build output directory.")

if(NOT PLUGIN_TARGET_PATH)
    set(PLUGIN_TARGET_PATH "$<TARGET_FILE_DIR:GaussianSplatting>")
    message(STATUS "PLUGIN_TARGET_PATH not set; deploying to default.")
else()
    file(TO_CMAKE_PATH "${PLUGIN_TARGET_PATH}" PLUGIN_TARGET_PATH)
    message(STATUS "PLUGIN_TARGET_PATH = ${PLUGIN_TARGET_PATH}")
endif()


# ---------------------------------------------------------------------------
# if we get more files, remember to put them here
# ---------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GaussianSplatting/src)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GaussianSplatting/shaders)

set(SOURCES
    ${SRC_DIR}/plugin_main.cpp
    ${SRC_DIR}/PLYReader.cpp
    ${SRC_DIR}/GaussianNode.cpp
    ${SRC_DIR}/GaussianDataNode.cpp
    ${SRC_DIR}/GaussianDrawOverride.cpp
    ${SRC_DIR}/GaussianRenderManager.cpp
    ${SRC_DIR}/GaussianSelection.cpp
)

set(HEADERS
    ${SRC_DIR}/GaussianData.h
    ${SRC_DIR}/PLYReader.h
    ${SRC_DIR}/GaussianNode.h
    ${SRC_DIR}/GaussianDataNode.h
    ${SRC_DIR}/GaussianDrawOverride.h
    ${SRC_DIR}/GaussianRenderManager.h
    ${SRC_DIR}/GaussianSelection.h
)

set(SHADERS
    ${SHADER_DIR}/gaussianDebug.fx
    ${SHADER_DIR}/Preprocessing.compute
)

# for .sln explorer
source_group("Source Files" FILES ${SOURCES})
source_group("Header Files" FILES ${HEADERS})
source_group("shaders" FILES ${SHADERS})



add_library(GaussianSplatting SHARED ${SOURCES} ${HEADERS} ${SHADERS})

set_source_files_properties(${SHADERS} PROPERTIES 
    HEADER_FILE_ONLY TRUE 
    VS_TOOL_OVERRIDE "None")

set_target_properties(GaussianSplatting PROPERTIES
    PREFIX ""
    SUFFIX ".mll"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_compile_definitions(GaussianSplatting PRIVATE
    NT_PLUGIN
    _CRT_SECURE_NO_WARNINGS
)

target_include_directories(GaussianSplatting PRIVATE
    ${SRC_DIR}
    "${MAYA_LOCATION}/include"
    "${DEVKIT_LOCATION}/include"
)

target_link_directories(GaussianSplatting PRIVATE
    "${MAYA_LOCATION}/lib"
    "${DEVKIT_LOCATION}/lib"
)

target_link_libraries(GaussianSplatting PRIVATE
    Foundation.lib
    OpenMaya.lib
    OpenMayaUI.lib
    OpenMayaAnim.lib
    OpenMayaFX.lib
    OpenMayaRender.lib
    Image.lib
    d3d11.lib
    d3dcompiler.lib
)

if(MSVC)
    target_compile_options(GaussianSplatting PRIVATE /W3 /MP /permissive- /Zc:__cplusplus)
    set_target_properties(GaussianSplatting PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE /export:initializePlugin /export:uninitializePlugin"
    )
endif()
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GaussianSplatting)

add_custom_command(TARGET GaussianSplatting POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PLUGIN_TARGET_PATH}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:GaussianSplatting>"
        "${PLUGIN_TARGET_PATH}/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADER_DIR}"
        "${PLUGIN_TARGET_PATH}/shaders"
    COMMENT "Deploying GaussianSplatting.mll + shaders to ${PLUGIN_TARGET_PATH}"
    VERBATIM
)


# ---------------------------------------------------------------------------
# 6. .mod 文件生成
#    格式:
#      + My3DGSPlugin 1.0 <module root>
#      plug-ins: .
# ---------------------------------------------------------------------------
set(MOD_FILE "${CMAKE_CURRENT_SOURCE_DIR}/3DGSPlugin.mod")

if(NOT EXISTS "${MOD_FILE}")

file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/3DGSPlugin_$<CONFIG>.mod"
    CONTENT "+ 3DGSPlugin 1.0 ${CMAKE_CURRENT_SOURCE_DIR}\nplug-ins: $<TARGET_FILE_DIR:GaussianSplatting>\n"
)

    message(STATUS "Wrote ${MOD_FILE}")
endif()

add_custom_command(TARGET GaussianSplatting POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/3DGSPlugin_$<CONFIG>.mod"
        "${CMAKE_CURRENT_SOURCE_DIR}/3DGSPlugin.mod"
    COMMENT "Updating Maya 3DGSPlugin.mod for $<CONFIG> mode..."
)
