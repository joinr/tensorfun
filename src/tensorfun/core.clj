(ns tensorfun.core
  (:require [tech.v3.datatype.functional :refer [- * + / ]]
            [tech.v3.tensor :as dtt]
            [fastmath.random :as fm.rand]
            [tensorfun.infix :as infix :refer [$= defop]]))

(defop '- 60 'tech.v3.datatype.functional/-)
(defop '+ 60 'tech.v3.datatype.functional/+)
(defop '/ 80 'tech.v3.datatype.functional//)
(defop '* 80 'tech.v3.datatype.functional/*)

(defn t [seq shape]
  (dtt/broadcast (dtt/->tensor seq :datatype :float32) shape)  )

(defn rt [shape]
  (dtt/->tensor (dtt/compute-tensor shape fm.rand/frand :float64)))

(defn read-tensor [contents]
  (let [tl (last contents)] ;;lame
    (if (vector? tl)
      (t (butlast contents) tl)
      (dtt/->tensor contents))))


#_(defop '<*> 80 'incanter.core/mmult)
#_(defop '<x> 80 'incanter.core/kronecker)


(def s [2000 2])

(def original
  (+
   (t [-6 -14] s)
   (*
    (t [14 18] s)
    (-> (dtt/compute-tensor s fm.rand/frand :float64)
        (dtt/->tensor)))))
;; #tech.v3.tensor<float64>[2000 2]
;; [[   -6.000    -11.60]

;;  [   -3.387     4.000]

;;  [    5.041     13.55]
;;  ...

;;  [     2179     517.8]

;;  [     6213 1.412E+04]

;;  [2.561E+04 2.424E+04]]

(def infixed
  (infix/$=
   (t [-6 -14] s) + (t [14 18] s)  * (rt s)))
;; #tech.v3.tensor<float64>[2000 2]
;; [[   -6.000    -7.512]

;;  [    4.683     4.000]

;;  [    11.13     14.71]
;;  ...

;;  [2.719E+04 3.442E+04]

;;  [1.463E+04 3.203E+04]

;;  [1.000E+04 1.249E+04]]

;;you'll get a complaint if you don't bind to a var in the
;;repl, since there's no print-dup defined yet.  tensors aren't readable.
;;We can't use them in macros at the moment, etc.
(def tagged
  (infix/$= #t[-6 -14 [2000 2]] + #t[14 18 [2000 2]]  * (rt s)))
;; #tech.v3.tensor<float64>[2000 2]
;; [[   -6.000    -3.168]

;;  [   -5.871     4.000]

;;  [    1.320     12.21]
;;  ...

;;  [2.174E+04 1.793E+04]

;;  [2.606E+04 2.051E+04]

;;  [2.503E+04 2.408E+04]]
