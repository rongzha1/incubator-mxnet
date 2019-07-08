# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

ifeq ($(USE_MKLDNN), 1)
	MKLDNN_SUBMODDIR = $(ROOTDIR)/3rdparty/mkldnn
	MKLDNN_BUILDDIR = $(MKLDNN_SUBMODDIR)/build
	MXNET_LIBDIR = $(ROOTDIR)/lib
ifeq ($(UNAME_S), Darwin)
	MKLDNN_LIBFILE = $(MKLDNNROOT)/lib/libmkldnn.0.dylib
	MKLDNN_LIB64FILE = $(MKLDNNROOT)/lib64/libmkldnn.0.dylib
else
	MKLDNN_LIBFILE = $(MKLDNNROOT)/lib/libmkldnn.so.0
	MKLDNN_LIB64FILE = $(MKLDNNROOT)/lib64/libmkldnn.so.0
endif
endif

.PHONY: mkldnn mkldnn_clean

mkldnn_build: $(MKLDNN_LIBFILE)

$(MKLDNN_LIBFILE):
	mkdir -p $(MKLDNNROOT)
	cmake $(MKLDNN_SUBMODDIR) -DCMAKE_INSTALL_PREFIX=$(MKLDNNROOT) -B$(MKLDNN_BUILDDIR) -DMKLDNN_ARCH_OPT_FLAGS="" -DMKLDNN_BUILD_TESTS=OFF -DMKLDNN_BUILD_EXAMPLES=OFF -DMKLDNN_ENABLE_JIT_PROFILING=OFF -DMKLDNN_USE_MKL=NONE -DCMAKE_BUILD_TYPE=Debug
	$(MAKE) -C $(MKLDNN_BUILDDIR) VERBOSE=1
	$(MAKE) -C $(MKLDNN_BUILDDIR) install
	mkdir -p $(MXNET_LIBDIR)
	if [ -f "$(MKLDNN_LIB64FILE)" ]; then \
		cp $(MKLDNNROOT)/lib64/libmkldnn* $(MXNET_LIBDIR); \
	else \
		cp $(MKLDNNROOT)/lib/libmkldnn*  $(MXNET_LIBDIR); \
	fi

mkldnn_clean:
	$(RM) -r 3rdparty/mkldnn/build
	$(RM) -r $(MKLDNNROOT)

ifeq ($(USE_MKLDNN), 1)
mkldnn: mkldnn_build
else
mkldnn:
endif
