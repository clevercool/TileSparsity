# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial

from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor

from utils import root_dir


def download_tf_model(arch_name, model_dir):
    full_model_dir = root_dir() / "downloads" / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    if not (full_model_dir / arch_name / "checkpoint").exists():
        download_arch(arch_name, str(full_model_dir / arch_name) + "/")


def download_all_tf_models():
    with ProcessPoolExecutor() as executor:
        list(
            executor.map(
                partial(download_tf_model, model_dir="model"),
                # for a complete list of architecture name supported, see
                # mmdnn/conversion/examples/tensorflow/extractor.py
                [
                    "vgg16",
                    # "vgg19",
                    # "inception_v1",
                    # "inception_v3",
                    # "resnet_v1_50",
                    # # "resnet_v1_152",
                    # "resnet_v2_50",
                    # "resnet_v2_101",
                    # "resnet_v2_152",
                    # # "resnet_v2_200",
                    # "mobilenet_v1_1.0",
                    # "mobilenet_v2_1.0_224",
                    # "inception_resnet_v2",
                    # "nasnet-a_large",
                    # "facenet",
                    # "rnn_lstm_gru_stacked",
                ],
            )
        )


def _single_thread_download(url, file_name):
    from six.moves import urllib

    result, _ = urllib.request.urlretrieve(url, file_name)
    return result


def download_file(
    url, directory="./", local_fname=None, force_write=False, auto_unzip=False
):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """

    if not os.path.isdir(directory):
        os.mkdir(directory)

    if not local_fname:
        k = url.rfind("/")
        local_fname = url[k + 1 :]

    local_fname = os.path.join(directory, local_fname)

    if os.path.exists(local_fname) and not force_write:
        print("File [{}] existed!".format(local_fname))
        return local_fname

    else:
        print("Downloading file [{}] from [{}]".format(local_fname, url))
        try:
            import wget

            ret = wget.download(url, local_fname, bar=None)
            print("")
        except Exception:
            ret = _single_thread_download(url, local_fname)

    if auto_unzip:
        if ret.endswith(".tar.gz") or ret.endswith(".tgz"):
            try:
                import tarfile

                tar = tarfile.open(ret)
                tar.extractall(directory)
                tar.close()
            except Exception:
                print("Unzip file [{}] failed.".format(ret))

        elif ret.endswith(".zip"):
            try:
                import zipfile

                zip_ref = zipfile.ZipFile(ret, "r")
                zip_ref.extractall(directory)
                zip_ref.close()
            except Exception:
                print("Unzip file [{}] failed.".format(ret))
    return ret


def download_arch(architecture, path="./"):
    if tensorflow_extractor.sanity_check(architecture):
        architecture_file = download_file(
            tensorflow_extractor.architecture_map[architecture]["url"],
            directory=path,
            auto_unzip=True,
        )
        if not architecture_file:
            return None

        if "ckpt" in tensorflow_extractor.architecture_map[architecture]["filename"]:
            tensorflow_extractor.handle_checkpoint(architecture, path)

        elif tensorflow_extractor.architecture_map[architecture]["filename"].endswith(
            "pb"
        ):
            tensorflow_extractor.handle_frozen_graph(architecture, path)

        else:
            raise ValueError(
                "Unknown file name [{}].".format(
                    tensorflow_extractor.architecture_map[architecture]["filename"]
                )
            )

        return architecture_file

    else:
        return None


if __name__ == "__main__":
    download_all_tf_models()
