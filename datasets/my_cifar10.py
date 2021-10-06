from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import CIFAR10
import torch
from research.utils import inverse_map

class MyCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs) -> None:
        cls_to_omit = kwargs.pop('cls_to_omit', None)
        super(MyCIFAR10, self).__init__(*args, **kwargs)
        if cls_to_omit is not None:
            assert cls_to_omit in self.classes
            label_to_omit = self.class_to_idx[cls_to_omit]
            self.classes.remove(cls_to_omit)
            del self.class_to_idx[cls_to_omit]

            for cls_str, cls_label in self.class_to_idx.items():
                if cls_label < label_to_omit:
                    continue
                elif cls_label > label_to_omit:
                    self.class_to_idx[cls_str] = cls_label - 1
                else:
                    raise AssertionError('cls_label={} should have been deleted by now'.format(cls_label))

            indices_to_omit = np.where(np.asarray(self.targets) == label_to_omit)[0]
            mask = np.ones(len(self.data), dtype=bool)
            mask[indices_to_omit] = False
            self.data = self.data[mask]
            self.targets = np.asarray(self.targets)[mask].tolist()

            # update targets
            for i, target in enumerate(self.targets):
                if target < label_to_omit:
                    continue
                elif target > label_to_omit:
                    self.targets[i] = target - 1
                else:
                    raise AssertionError('target={} should have been deleted by now'.format(target))
        self.idx_to_class = inverse_map(self.class_to_idx)
        self.idx_to_glove_vec = self.set_glove_vecs()

    @staticmethod
    def parse_vec(s: str):
        return np.asarray(list(map(float, s.split())))

    def set_glove_vecs(self):
        embs = np.empty((100, 200), dtype=np.float32)

        embs[0] = self.parse_vec('-0.42252 -0.72499 0.3823 -0.28675 -0.070732 1.082 0.61925 -0.51744 -0.24192 0.36525 -0.10519 0.68813 -0.82839 0.0121 -0.30335 0.057322 0.077832 0.11161 0.46033 -0.21916 -0.049768 -0.24293 0.12415 -0.40696 0.32383 1.0217 0.62564 -0.75066 -0.41027 -0.0758 -0.1808 -0.027986 0.21466 -1.1386 0.20759 0.67844 -0.60843 0.28039 1.0015 0.014468 0.2675 -0.10874 -0.23052 -0.83247 0.2413 -0.11418 -0.31517 -0.28662 0.067465 0.17122 0.16358 -0.38727 -0.33752 0.15207 0.071406 -0.23285 -0.39292 0.79661 -0.01181 -0.61611 0.42596 -0.024823 0.51229 -0.1942 -0.31514 -0.9923 0.26809 -0.16498 0.20328 -0.21459 -0.70433 -0.0017985 -0.65342 -0.85474 0.161 -0.71959 -0.50075 -0.18926 0.31129 0.90581 0.58413 0.87044 -0.056666 -0.26441 0.29036 0.07847 0.026343 0.3536 -1.1024 0.4081 0.26188 -0.20925 -0.728 -0.04421 -0.21305 -0.2336 -0.33843 -0.27006 -0.81843 -0.19834 0.58124 0.039614 -0.90533 0.39462 -0.35865 -0.47045 0.22981 -0.044953 0.28625 -0.14308 -0.31557 -0.015199 -0.28968 -0.28257 -0.72873 -0.13707 -0.0014256 -0.44722 -0.14099 -0.062103 0.53414 -0.18197 -0.13406 -0.41105 -0.39153 0.73264 0.031486 0.3796 0.40439 0.37544 0.49086 0.38665 0.095826 0.2573 -0.47709 -0.5425 0.19142 0.66534 -0.26036 0.044465 -0.1965 0.21443 0.090587 0.48187 0.063059 0.10099 0.23694 -0.16066 -0.39295 -0.62392 1.2988 -0.2949 -1.8037 0.32934 -0.11134 0.0236 0.29623 -0.39351 0.058452 -0.37467 -0.029277 0.073365 0.3801 0.67572 0.10034 -0.27386 -0.58898 0.18683 0.029444 0.20757 0.01653 -0.4761 0.15124 -0.24604 0.064738 0.22999 -0.80299 0.20186 -0.012943 0.80957 0.25185 -0.28367 -0.0093086 0.2747 -0.91049 0.24138 0.31127 -0.084327 0.15578 -0.23792 0.74639 -0.24335 -0.084517 -0.072658 0.027183 0.083656 0.10962 0.025677 0.26856 0.049582')
        # use "car" instead of "automobile"
        embs[1] = self.parse_vec('-0.023756 -0.6095 -0.64204 0.21877 0.46728 0.18328 -0.017327 -0.1671 0.15519 -0.19869 0.58117 0.40394 -0.39322 -0.14633 -0.14179 0.015474 0.11165 -0.10333 -0.20328 -0.071406 0.12644 -0.26139 -0.36218 -0.67246 -0.34604 0.59822 -0.17553 -0.031497 0.11128 -0.3225 0.061777 0.38997 -0.33846 -0.1767 -0.082802 0.41319 -0.47078 0.48865 0.74484 0.24344 0.43444 0.34383 -0.63643 0.41448 -0.38013 -0.16224 0.41776 -0.045915 0.76219 0.055854 0.80065 0.22815 -0.95708 -0.064152 -0.25136 0.030722 -0.56599 0.13781 0.093393 -0.83462 0.32205 -0.065024 0.86411 -0.054507 0.19187 -0.39785 0.16377 0.57524 -0.37361 -0.72036 -0.48547 0.18768 -0.2428 -0.0031741 -0.43129 0.21333 -0.36452 0.15536 -0.18761 0.43804 0.66989 0.1977 -0.48026 0.17955 -0.26623 0.3866 0.37762 0.33181 -0.29401 0.089559 -0.1417 0.090185 0.23631 0.05726 0.49807 0.5556 0.0085019 -0.19751 -0.99868 -0.12837 0.72538 -0.21058 -0.17776 0.54406 -0.51257 -0.30398 0.5172 -0.4982 0.72498 -0.13728 -0.15657 0.48735 -0.12313 -0.44957 0.10629 0.13345 -0.71389 -0.41793 -0.77205 0.70404 0.35033 -0.33719 -0.23397 -0.18326 -0.36967 0.76203 0.23946 0.85417 0.069386 -0.19864 0.38917 -0.12225 -0.34538 0.062926 -0.31898 0.17836 -0.4046 0.38409 -0.20409 0.35095 -0.42669 -0.06645 0.2125 0.14951 -0.23864 0.1338 0.11083 0.21279 -0.0037618 -0.13022 0.21465 -0.51508 -4.7217 0.15789 0.26162 -0.15878 0.012484 -0.13879 0.40189 -0.49206 0.35261 0.62121 0.37681 0.54427 0.06366 -0.3226 -0.47194 -0.6409 -0.16708 -0.067091 0.21019 0.52271 -0.51378 -0.45009 0.77929 -0.033527 0.34275 0.15728 0.22613 1.0059 0.091323 0.025024 0.1937 0.17346 0.35938 -0.59598 0.52244 -0.32664 0.23388 0.29734 -0.1782 -0.58709 0.58139 -0.39022 -0.17797 0.02756 -0.2737 0.00032772 0.3212 0.31734')
        embs[2] = self.parse_vec('0.050286 -0.40342 -0.085693 -0.11261 -0.40626 0.27764 0.28587 -0.036815 0.29082 0.53717 -0.096179 0.20294 -0.52494 -0.42556 -0.020042 0.59147 0.60556 -0.096592 0.078125 -1.009 -0.48508 0.26272 -0.36493 -0.72437 0.044094 0.46839 0.22695 0.080163 -0.18623 0.49568 -0.067437 0.29948 -0.36965 -0.73587 -0.033697 0.35647 -0.13801 0.42026 -0.064175 -0.35642 -0.40864 0.081728 0.1202 -0.45304 0.35192 -0.16238 -0.40587 0.28837 0.72754 0.5276 -0.12201 -0.18372 0.36878 0.46526 0.32681 -0.56752 -0.50191 0.60814 0.57881 0.0227 0.23608 0.035366 0.16645 -0.028746 -0.13858 -0.42193 0.42848 -0.011398 0.32289 0.204 -0.34057 0.30971 -0.5685 -0.85169 -0.12805 -0.3842 -0.11821 0.050055 0.50502 0.58767 1.0039 0.3996 -0.027687 0.17466 -0.22844 0.12718 -0.51194 -0.45218 -0.20525 0.055035 0.27 -1.0207 -1.1003 -0.51314 -0.35455 -0.13669 -0.17903 0.10799 -0.24093 0.66859 -0.13704 0.50379 -0.065461 0.15555 -0.51893 0.62364 -0.52682 0.16933 -0.44093 -0.090353 -0.84958 0.42558 -0.31874 -0.38313 0.39895 -0.067433 1.0144 -0.17431 -0.063368 -0.60363 0.20053 0.13679 -0.024741 0.47469 -0.77892 -0.28663 -0.27192 -0.67562 0.28207 0.1935 0.063162 0.73112 0.072682 0.51456 -0.55077 -0.25402 -0.077662 0.035238 -0.32021 -0.33759 -0.24357 0.035842 0.81423 -0.3508 0.18006 -0.049245 0.12888 -0.16803 -0.3665 0.63389 -0.13232 -0.54769 -3.4213 -0.38828 -0.24938 -0.41294 -0.2727 -0.3304 0.23315 -0.52551 0.21471 -0.38583 -0.30177 0.30061 -0.33541 -0.60107 0.23551 -0.80369 -0.13737 -0.1429 0.16166 0.32293 -0.12294 0.16138 -0.093296 0.14234 0.27728 0.036312 -0.19796 0.1936 -0.46891 0.82351 -0.53899 -0.24703 0.049887 0.54725 0.009746 0.57974 -0.0091502 -0.34196 0.026213 0.19177 0.5079 0.16918 0.6699 0.4473 -0.61384 -0.015805 -0.42108 -0.087537')
        embs[3] = self.parse_vec('0.14557 -0.47214 0.045594 -0.11133 -0.44561 0.016502 0.46724 -0.18545 0.41239 -0.67263 -0.48698 0.72586 -0.22125 -0.20023 0.1779 0.67062 0.41636 0.065783 0.48212 -0.035627 -0.47048 0.077485 -0.28296 -0.49671 0.337 0.71805 0.22005 0.12718 0.067862 0.40265 -0.01821 0.78379 -0.52571 -0.39359 -0.56827 -0.15662 -0.084099 -0.20918 -0.066157 0.25114 -0.40015 0.1593 0.17887 -0.3211 0.09951 0.52923 0.48289 0.14505 0.44368 0.17365 0.3635 -0.51496 -0.12889 -0.19713 0.18096 -0.011301 0.84409 0.98606 0.83535 0.3541 -0.23395 0.3551 0.41899 -0.054763 0.22902 -0.19593 -0.57777 0.29728 0.33972 -0.31119 -0.32498 -0.42557 -0.70302 -0.72515 -0.29349 0.49964 -0.32889 0.24359 0.13243 0.31164 1.2156 0.31241 -0.23794 0.38422 -0.321 -0.28756 -0.20047 0.34454 -0.64929 0.28021 0.060203 0.053618 -0.13341 0.2451 0.18639 -0.0016346 -0.066883 0.077845 -0.085217 0.75257 0.76264 -0.053318 0.071056 0.30552 -0.43411 -0.19361 -0.10493 -0.53732 -0.239 -0.47298 -0.029825 -0.20206 -0.48945 -0.13616 0.49622 0.20743 -0.077396 -0.34304 0.0062387 -0.0065902 -0.24729 -0.013859 -0.079919 0.43452 0.23415 0.17995 0.13236 -0.22717 -0.55278 0.042005 0.21937 0.42042 0.43639 -0.58305 -0.118 0.15379 -0.29596 -0.46251 0.52593 0.10471 -0.19973 -0.028228 0.49974 -0.58053 -0.51416 0.21325 -0.38394 -0.00059821 0.16525 -0.055993 -0.4008 -0.05483 -3.8842 -0.022136 -0.46989 0.23502 0.081298 0.83091 0.47251 0.074057 0.15737 0.065809 -0.26756 0.1947 -0.63597 -0.59914 -0.21369 0.011718 -0.25464 -0.19629 0.18017 0.59031 0.0062176 0.51122 0.36601 -0.27381 -0.11342 0.21195 0.43099 -0.43837 0.12842 0.39312 -0.19492 0.056414 0.54343 0.13678 -0.71087 0.38758 -0.0078956 -0.32383 0.064193 -0.22329 0.071366 -0.30966 -0.46142 0.29545 -0.49186 0.24053 -0.46081 -0.077296')
        embs[4] = self.parse_vec('-0.88888 -0.8283 -0.26092 -0.1878 0.29118 0.53808 0.35139 0.13929 0.12715 -0.36606 0.11742 0.85401 0.31036 0.58252 -0.10211 0.16705 0.21117 -0.97241 0.21983 -0.73822 0.3767 0.53084 -0.027001 0.1418 0.0030691 1.3255 0.032468 0.76395 -0.20878 0.55375 0.31824 0.48514 -0.10506 0.0061221 -0.33417 0.71345 -0.67179 -0.1916 0.040424 -0.56084 0.21238 0.3659 -0.59084 0.40843 0.32561 0.054806 0.60712 -0.094207 1.1577 0.029936 -0.64206 -0.63351 -0.087974 0.28289 0.83288 -0.79945 -0.14793 0.91633 0.67526 0.54034 -0.67172 0.39372 -0.023389 -0.17461 -0.10204 0.34649 -0.18225 -0.11211 -0.01391 -0.40143 -0.59917 0.052459 -0.45205 -0.15403 0.10181 0.57976 -0.47397 -0.29409 0.062632 -0.68245 0.40346 0.16259 -0.12952 0.65635 0.27925 0.11619 -0.2168 0.4119 -0.060018 0.39624 0.31234 0.12689 -0.66612 -0.39723 -0.1087 0.29764 -0.43254 0.08048 -0.027164 0.41759 -0.00031392 0.30644 -0.21959 -0.43071 -0.39769 -0.55354 -0.24261 -0.26165 -0.34768 -0.44018 -0.63894 -0.38801 -0.26602 0.21619 0.53052 0.16398 -0.48921 -0.12992 -0.67551 -0.31028 0.095231 0.087434 -0.005276 -0.09389 0.12179 -0.17634 -0.38522 -0.0074983 0.89795 -0.10206 0.87212 0.023518 -0.25015 0.36739 -0.86754 -0.029043 0.048827 0.088942 -0.18248 0.13662 0.5255 0.36528 0.30313 -0.36591 -0.4889 -0.12098 0.10393 0.14775 -0.05941 0.2426 -0.12164 0.19213 -2.2862 0.043635 0.088264 -0.39852 -0.24304 0.32002 -0.049949 0.51055 0.86949 -0.385 -0.10674 0.052262 -0.19739 -0.36232 -0.45759 -0.81358 -0.023949 -0.72994 -0.30646 -0.036489 -0.45232 0.17008 0.29783 -0.31853 -0.00482 -0.11042 0.11359 -0.072519 -0.063388 0.32953 0.34181 -0.16518 0.0099182 0.68158 0.72981 -0.0050325 -0.47872 0.44239 -0.29148 0.0054185 -0.37443 0.6663 -0.72571 0.57017 0.14228 -0.096767 -0.099185 -0.10899')
        embs[5] = self.parse_vec('-0.49586 -0.59369 -0.107 0.05593 -0.24633 -0.14021 0.63707 0.024992 0.25119 -0.55602 -0.37298 0.60131 -0.35971 -0.096752 0.18511 0.58992 0.47578 -0.16833 0.67079 -0.29472 0.069403 0.05334 -0.36154 -0.12883 0.27814 0.87467 0.12119 0.78215 -0.50617 0.28794 0.14213 0.83281 -0.27079 -0.28813 -0.67607 0.17991 -0.11046 -0.063062 -0.56297 0.36639 0.11009 0.2965 -0.12457 -0.11112 -0.24293 0.53344 0.75589 0.078154 0.91641 0.20878 0.01236 -0.71199 0.19085 -0.5199 -0.14181 0.078136 0.44157 1.0958 0.59009 0.35117 0.021684 0.1073 0.19942 -0.26355 0.084024 -0.32073 -0.24306 0.44821 0.14432 -0.063988 -0.15013 -0.33644 -0.67873 -0.64554 0.10706 0.64709 -0.20094 0.064682 0.035356 0.029288 0.99793 0.34343 -0.019469 0.70635 -0.54329 -0.057843 0.12624 -0.18132 0.099001 0.4478 -0.2641 -0.37506 -0.11238 -0.011805 0.33187 0.45295 0.1682 0.18379 0.29457 0.98963 0.5394 -0.0025833 -0.10989 0.30163 0.34495 -0.2275 -0.21093 -0.79685 0.29833 -0.64644 -0.18653 0.31771 0.061874 -0.44503 0.34052 0.5552 0.017743 -0.33609 0.18478 0.392 -0.44685 -0.2591 -0.4929 0.61712 -0.24546 0.15348 0.19796 0.041105 0.030167 0.13735 0.29154 0.079533 0.53594 -0.61848 0.082946 -0.43806 -0.16041 -0.44336 0.065162 0.29823 -0.13321 0.55445 0.29978 -0.63209 -0.45078 0.1534 -0.31124 0.258 0.062033 0.047879 0.37758 -0.007643 -4.328 0.65362 -0.45488 -0.4565 0.23566 1.0171 0.53344 -0.025861 0.067191 0.60342 -0.56511 0.57175 -0.47311 -0.43066 -0.13385 0.011506 -0.32674 -0.47726 0.010775 0.49053 -0.11302 0.23358 0.098286 -0.55746 0.096976 0.036503 0.41838 -0.22967 0.12346 0.23573 -0.17653 0.03863 0.62339 -0.083598 -0.62161 0.11059 0.11316 -0.26833 0.023406 -0.018887 -0.63446 -0.16513 -0.16886 0.087242 -0.10353 0.06788 -0.20546 0.17962')
        embs[6] = self.parse_vec('-0.10524 -0.24913 0.52241 0.15184 -0.19537 0.11306 0.88643 -0.1914 -0.15109 -1.0077 -0.133 0.82179 -0.57923 0.077041 -0.17322 0.85412 -0.035148 -0.21483 0.53501 -0.14371 0.11799 -0.01616 -0.15576 -0.21612 0.88443 0.78456 0.01876 0.040336 -0.099669 0.12756 0.52273 0.62248 -0.48771 -0.79751 -0.61862 0.54606 0.35053 0.31885 0.14881 0.00096039 0.7272 0.2406 -0.40315 0.56772 0.18635 0.54332 -0.033322 0.2073 1.0884 0.45316 0.23937 -0.61737 0.15905 -0.11814 0.28906 -0.84945 0.46633 0.0715 0.1423 0.80191 -0.19417 0.33199 0.4975 -0.20668 -0.58879 -0.49594 0.11781 -0.046176 0.085102 -0.099519 0.080947 0.35439 0.0096015 -1.1062 0.15469 -0.5371 -0.40395 0.32594 0.055478 -0.3327 0.44476 0.46102 -0.14429 0.1887 -0.18255 0.51218 0.72426 -0.43208 -0.92449 0.51202 0.24498 -0.13528 -0.13124 0.40501 0.29041 -0.031704 0.30775 -0.39611 -0.23103 1.0397 -0.18068 0.26069 -0.35832 -0.067333 -0.26544 0.16485 -0.50203 0.013022 -0.53152 -0.23262 -0.2995 0.10268 0.0081069 -0.48369 0.33022 -0.29557 -0.4956 0.037559 0.031966 0.36481 0.073253 -0.47276 -0.16962 -0.12635 -0.61112 0.053608 -0.40487 -0.49995 0.038738 0.30166 0.98378 0.27356 0.15385 -0.075244 0.34844 0.42685 -0.043121 -0.091476 -0.42405 0.03645 -0.13394 0.64432 -0.015429 -0.19813 -0.020037 0.36982 -0.065552 0.16153 0.14069 0.014353 0.098342 0.47037 -2.169 -0.37849 -0.30882 0.14805 0.0524 0.21195 -0.29152 -0.37244 -0.033199 -0.78176 0.51217 0.42594 -0.077918 -0.6639 0.15198 -0.11359 0.24057 -0.083828 -0.0015019 0.18238 -0.12708 0.44399 0.30008 -0.094542 -0.29812 0.0014999 -0.081555 -0.47116 -0.48297 0.18062 0.086784 -0.10048 0.20013 0.46957 0.54502 0.059997 -0.64538 0.17187 0.62724 -0.075442 0.40395 0.52707 0.58531 0.03689 -0.34729 0.33194 -0.54962 0.27767')
        embs[7] = self.parse_vec('-0.8107 -0.2135 0.57229 0.38901 -0.53731 0.076275 0.80555 -0.64481 0.58122 -0.003714 0.15482 0.5188 -0.73224 -0.17708 0.37883 1.0903 0.39686 -0.38992 0.45664 -0.31646 0.49369 -0.16371 -0.45948 -0.21822 0.34105 0.96526 0.25932 0.12078 0.012586 0.084278 0.50996 0.27742 -0.15154 -0.13721 -0.098856 0.12999 -0.41539 0.21986 -0.27817 -0.1278 0.1805 -0.71333 0.3577 0.42558 0.25589 0.443 0.36289 0.17151 1.0117 0.74856 0.26782 -0.029225 -0.36808 -0.13197 0.51501 0.13333 0.0058557 0.80578 -0.0721 0.70669 -0.50893 1.2565 0.20282 -0.13758 -0.5108 -0.34195 -0.24551 0.53538 0.2398 -0.30907 -0.20728 -0.82592 -0.34368 0.017876 0.092939 0.049257 -0.43085 -0.13684 0.019521 -0.20954 0.58053 -0.18977 -0.28645 0.44486 -0.5442 0.708 0.46365 0.086484 -0.042811 0.04067 -0.26089 -0.4174 -0.35112 -0.45257 0.27432 0.42729 0.4371 0.31975 0.017235 0.42254 -0.053444 -0.16006 -0.31785 0.33874 -0.23682 -0.34646 -0.30786 -0.55616 -0.045204 0.012021 -0.63051 0.3996 -0.29002 0.0079054 0.047329 0.5004 0.060087 -0.2037 0.12378 0.24339 -0.38377 -0.50928 -0.1049 0.14504 -0.39883 -0.24158 -0.33095 0.20819 0.81785 -0.34484 0.25812 0.017235 0.25583 -0.096405 0.16331 0.12816 -0.1257 0.11052 -0.19591 0.26462 -0.093251 0.74641 0.37195 -0.19395 -0.26052 -0.36437 0.46078 0.22374 -0.15367 0.3202 0.19659 -0.18048 -3.2003 0.24416 -0.36079 -0.022701 -0.10411 0.57065 0.20385 0.020388 0.78644 0.55647 -0.1408 -0.11196 -0.50173 -0.38527 -0.2307 0.062547 -0.54328 -0.56776 0.38209 0.10156 -0.16395 0.35198 0.55722 -0.34555 0.017989 -0.040839 0.28383 -0.049434 0.11944 0.086508 0.4774 0.073957 -0.23412 0.29014 -0.14949 -0.2585 -0.29038 1.0173 0.59803 -0.083486 0.30558 0.47593 0.026809 0.090965 0.052627 0.074359 -0.36702 0.20615')
        embs[8] = self.parse_vec('-0.046384 -0.1674 0.13209 0.27851 -0.3038 -0.18791 0.67359 0.43768 0.7281 -0.32125 0.37458 -0.65296 -0.39845 0.68764 -0.76904 0.42809 -0.17888 0.26656 -0.41712 -0.59264 0.14285 -0.72536 0.14494 -0.3611 -0.25282 0.9417 0.52213 0.030727 0.3491 0.074366 0.12063 -0.46192 0.32909 -0.35875 -0.78685 -0.20006 -0.015433 -0.29698 0.1588 -0.23569 -0.11503 -0.77778 -0.45981 -0.17776 -0.56484 -0.070167 0.24729 -0.93119 0.71711 0.24224 0.99186 0.69913 0.22442 0.20017 -0.10483 0.10171 -0.12559 0.66996 0.084915 0.14392 0.083959 0.20461 0.53045 -0.89678 -0.26585 0.12609 -0.15158 0.0030346 -0.28946 -0.66158 -0.010883 0.064274 0.58277 -0.26354 0.73509 -0.41874 -0.82432 0.85091 -0.030977 -0.52292 0.84651 0.0030065 0.41642 0.71237 0.12576 -0.51626 -0.056885 0.26866 -0.58667 0.2174 -0.11079 0.10996 -0.34554 0.078617 -0.2944 -0.1506 -0.45091 0.12881 0.30986 0.13821 0.52604 0.27173 -0.71395 0.18905 -0.81359 0.36882 -0.31025 0.29672 0.38817 -0.27687 -0.49648 0.28523 -0.3831 -0.95998 0.17105 -0.4048 0.034503 -0.40979 0.38877 0.13077 0.40904 1.0764 -0.27713 -0.020123 0.059802 0.88358 -0.13797 0.38392 -0.069404 0.4747 -0.37298 0.060361 -0.3811 0.64165 0.015458 0.54536 0.30043 0.30915 -0.14612 -0.26519 -1.0387 -0.17801 0.092653 -0.039417 -0.02523 -0.20054 -0.22877 -0.23474 -0.099949 0.48674 0.59772 -0.90361 -3.1459 -0.28061 0.75944 0.22351 0.064689 0.055654 -0.14251 0.44982 -0.28356 -0.038955 0.041129 -0.16093 0.72469 -0.15257 0.10329 0.0713 -0.80154 -0.24854 -1.3841 -0.16126 -0.41553 0.013537 0.182 -0.14519 -0.46814 0.34555 -0.13577 0.11977 0.19675 -0.29271 -0.11296 0.012884 0.062272 0.46016 0.63468 0.3659 -0.004723 -0.46667 0.41385 0.52496 0.048587 -0.35789 0.93803 0.19896 0.63798 -0.4019 0.53144 -0.022184')
        embs[9] = self.parse_vec('-0.24213 -0.51403 -0.51578 -0.3406 0.41113 -0.17706 0.39752 -0.53812 -0.26475 -0.40181 0.36399 0.47676 -0.52686 0.1865 0.044458 -0.403 0.67248 0.39037 0.082186 -0.28023 -0.12542 -0.53253 -0.74728 -0.60064 -0.52848 1.0082 -0.40989 0.25493 0.035693 -0.76599 -0.083746 0.42385 -0.096547 -0.35689 0.28786 0.33861 -0.6818 0.37552 0.32567 0.60148 0.51642 0.033812 -1.0883 0.16184 0.22601 0.18564 0.64249 -0.31606 0.62122 0.24061 0.4768 -0.06472 -0.52014 0.075044 -0.17287 -0.656 -0.1495 0.27403 0.0042714 -0.1294 0.084415 0.048958 0.23673 -0.18547 0.044316 -0.62387 -0.26171 -0.025662 -0.02504 -1.0354 -0.2403 -0.05565 -0.51739 0.23079 -0.30098 0.59234 -0.42412 -0.18262 0.16775 0.27964 0.2425 -0.29826 -0.52536 0.45102 -0.49331 0.38804 -0.078744 0.11839 -0.04422 0.40894 0.080419 0.46671 -0.18263 -0.22276 0.3776 0.60906 0.15127 0.47851 -0.44776 0.26126 0.65134 -0.2067 0.00050787 -0.12693 -0.12946 -0.47426 0.089425 -0.35967 0.50107 -0.039612 -0.23707 0.58465 -0.082157 -0.13513 0.12528 0.17546 -0.6148 -0.23756 -0.60866 0.91155 -0.10848 0.061624 -0.36944 0.1166 -0.45904 0.37417 -0.0051873 1.2031 0.031788 0.14029 0.46157 -0.22442 0.1772 0.3102 -0.56272 0.23431 -0.13753 0.23723 -0.046944 -0.48279 -0.86001 0.22843 0.35981 0.17068 -0.91521 -0.76246 0.34586 -0.40751 0.34906 -0.092519 0.70924 -0.29729 -3.2132 0.027102 0.70242 -0.43226 -0.14798 0.154 0.26714 -0.14026 0.17793 0.08782 0.52942 0.20536 0.32086 -0.31948 -0.66555 -0.54011 -0.35877 -0.4519 -0.11377 0.097477 -0.83047 -0.11043 0.66162 -0.20076 0.14835 0.71861 0.61731 1.1303 -0.2095 -0.62456 0.23265 -0.12174 0.1635 -0.33432 0.61246 0.16256 0.36239 0.38967 -0.23486 -0.02727 0.12782 0.025456 -0.13723 -0.39873 0.1555 0.37518 0.1981 0.31379')

        return embs

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if type(img) != torch.Tensor:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
