# ===================================================================================
#  cvsba CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE(cvsba REQUIRED )
#    TARGET_LINK_LIBRARIES(MY_TARGET_NAME )
#
#    This file will define the following variables:
#      - cvsba_LIBS          : The list of libraries to links against.
#      - cvsba_LIB_DIR       : The directory where lib files are. Calling LINK_DIRECTORIES
#                                with this path is NOT needed.
#      - cvsba_VERSION       : The  version of this PROJECT_NAME build. Example: "1.2.0"
#      - cvsba_VERSION_MAJOR : Major version part of VERSION. Example: "1"
#      - cvsba_VERSION_MINOR : Minor version part of VERSION. Example: "2"
#      - cvsba_VERSION_PATCH : Patch version part of VERSION. Example: "0"
#
# ===================================================================================
INCLUDE_DIRECTORIES("/usr/local/include")
SET(cvsba_INCLUDE_DIRS "/usr/local/include")

LINK_DIRECTORIES("/usr/local/lib")
SET(cvsba_LIB_DIR "/usr/local/lib")

SET(cvsba_LIBS /usr/lib/x86_64-linux-gnu/libblas.so;/usr/lib/x86_64-linux-gnu/liblapack.so;/usr/lib/x86_64-linux-gnu/libf2c.so;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_datasets;opencv_dnn_objdetect;opencv_dnn_superres;opencv_dpm;opencv_face;opencv_freetype;opencv_fuzzy;opencv_hdf;opencv_hfs;opencv_img_hash;opencv_line_descriptor;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_quality;opencv_reg;opencv_rgbd;opencv_saliency;opencv_shape;opencv_stereo;opencv_structured_light;opencv_superres;opencv_surface_matching;opencv_text;opencv_tracking;opencv_videostab;opencv_viz;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto;-lgomp cvsba) 

SET(cvsba_FOUND YES)
SET(cvsba_FOUND "YES")
SET(cvsba_VERSION        1.0.0)
SET(cvsba_VERSION_MAJOR  1)
SET(cvsba_VERSION_MINOR  0)
SET(cvsba_VERSION_PATCH  0)
