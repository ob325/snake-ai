(ns snake-ai.core
  (:require [quil.core :as q]
            [quil.middleware :as m]
            [clojure.core.matrix :as mtrx]))

(def board-width 12)
(def segment-width (/ 400 (inc board-width)))
(def board-pixels (* segment-width board-width))
(def population 512)
(def n-inputs 16)
(def n-hidden 12)
(def n-output 4)
(def n-survivors 64)
(def weight-range [-3 3])
(def all-board-coords (into #{} (for [x (range board-width)
                                      y (range board-width)] [x y])))

;; neural network 
(def bool
  {true 1 false 0 nil 0})

(def output-map
  {0 [1 0] 1 [-1 0] 2 [0 1] 3 [0 -1]})

(defn relu [x]
  (if (< x 0) 0 x))

(defn sigmoid [x]
  (/ 1 (inc (Math/exp (* -1 x)))))

(defn step [x]
  (if (< x 0) 0 1))

(defn idx-max [x]
  (first (apply max-key second (map-indexed vector x))))

(defn random-weight []
  (let [[a b] weight-range] (+ a (rand (- b a)))))

(defn initialize-genome []
  (let [len (+ (* (inc n-inputs) n-hidden) (* (inc n-hidden) n-output))]
    (take len (repeatedly random-weight))))

(defn mutate [g]
  (map #(* % (if (< (rand) 0.95) 1.0 (rand-nth [0.5 1.5]))) g))

(defn encode-genome [genome]
  (let [[t1 t2] (split-at (* (inc n-inputs) n-hidden) genome)]
    (list (partition (inc n-inputs) t1)
          (partition (inc n-hidden) t2))))

(defn crossover [a b]
  (mutate (map rand-nth (partition 2 (interleave a b)))))

(defn forward-propagate [i g]
  (let [[t1 t2] g
        h1-activations (as-> (cons 1 i) x
                         (mtrx/mmul t1 x)
                         (map sigmoid x))
        outputs (as-> h1-activations x
                  (cons 1 x)
                  (mtrx/mmul t2 x)
                  (map sigmoid x)
                  (vec x))]
    {:inputs i
     :h1-activations h1-activations
     :outputs outputs}))

 
(defn features [s]
  "Extract neural network input features from current state"
  (let [[head-x head-y] (last (:snake s)) 
        [apple-x apple-y] (:apple s)      
        diagonals [(< head-x apple-x)     
                   (> head-x apple-x)     
                   (< head-y apple-y)
                   (> head-y apple-y)]
        bounds [(zero? head-x) (zero? head-y) (= board-width head-x) (= board-width head-y)]
        apple-ortho [(and (= head-x apple-x) (> head-y apple-y))
                     (and (= head-x apple-x) (< head-y apple-y))
                     (and (= head-y apple-y) (> head-x apple-x))
                     (and (= head-y apple-y) (< head-x apple-x))]                          
        orthogonal-to-head (mapv #(mapv + (last (:snake s)) %) [[0 1] [0 -1] [1 0] [-1 0]])
        body-to-head (mapv (fn [x] (some #(= x %) (:snake s))) orthogonal-to-head)]
    (concat (map bool diagonals)
            (map bool bounds)
            (map bool apple-ortho)
            (map bool body-to-head))))

;; gameplay
(defn initial-energy [g] (+ 100 (* 20 g)))

(defn new-apple
  "Returns coords for a new apple not overlapping snake's body"
  [body]
  (rand-nth (remove (into #{} body) all-board-coords)))

(defn updated-snake [curr dir apple]
  (let [new-head (mapv + (last curr) dir)]
    (if (not= new-head apple)
      (vec (rest (conj curr new-head)))
      (conj curr new-head))))

(defn out-of-bounds? [s]
  (some #(or (> % board-width)
             (< % 0))
        (last (:snake s))))

(defn out-of-energy? [s] (= 0 (:energy s)))

(defn alive? [s]
  (and (not (some #(< % 0) (last (:snake s))))
       (not (some #(> % board-width) (last (:snake s))))
       (>= (:energy s) 0)
       (= (count (:snake s)) (count (distinct (:snake s))))))

(defn new-game-state [s]
  (as-> (update s :individual inc) x
    (assoc x :snake [[5 5] [5 4] [5 3]])
    (assoc x :energy (initial-energy (quot (:individual x) population)))
    (assoc x :apple (new-apple (:snake x)))
    
    (assoc x :pool (if (zero? (mod (:individual x) population))
                     (as-> (:scores x) k
                       (sort-by :score #(compare %2 %1) k)
                       (map :genome k)
                       (take n-survivors k)
                       (partition 2 1 k)
                       (map #(crossover (first %) (second %)) k))
                     (:pool x)))
    
    (assoc x :scores (if (zero? (mod (:individual x) population))
                       []
                       (conj (:scores x) {:score (count (:snake s))
                                          :genome (flatten (flatten (:genome x)))})))
      
    
    (assoc x :genome (if (> (:individual x) (dec population))
                       (encode-genome (rand-nth (:pool x)))
                       (encode-genome (initialize-genome))))))

(defn score-graph [data]
  (let [n (count data)
        x (map #(q/map-range % 0 n 0 350) (range n))
        y (map #(q/map-range % 0 (apply max data) 0 125) data)
        coords (partition 2 (interleave x y))]
    (q/with-translation [430 80]
      (doseq [[x y] coords]
        (q/rect x (- 125 y) 2 y)))))
  

;; quil functional-mode
(defn setup []
  (q/color-mode :hsb)
  (let [g (encode-genome (initialize-genome))]
    {:individual 0
     :snake [[5 5] [5 4] [5 3]]  ;;[(rand-nth (vec all-board-coords))]
     :energy (initial-energy 20)
     :apple (new-apple [[4 4]])
     :genome g
     :high-score 0
     :high-score-by-gen []
     :high-score-this-gen 0
     :scores []
     :pool []
     :nnet []}))

(defn update-state [state]
  (let [nnet (forward-propagate (features state) (:genome state))
        d (output-map (idx-max (:outputs nnet)))
        head (last (:snake state))]
    (as-> (assoc state :snake (updated-snake (:snake state) d (:apple state))) s
      (assoc s :nnet nnet)
      (assoc s :high-score-this-gen (max (:high-score-this-gen s) (count (:snake s))))
      (update s :energy dec)
      (assoc s :high-score (max (count (:snake s)) (:high-score s)))
      (if (= head (:apple state)) (assoc s :apple (new-apple (:snake s))) s)
      (if (alive? s) s (new-game-state s)))))

(defn draw-state [state]
  (q/frame-rate 256)
  (q/background 16)

  (q/fill 70 255 255)
  (q/text (format "Generation: %s" (quot (:individual state) population)) 430 10)
  (q/text (format "Individual: %s" (mod (:individual state) population)) 430 20)
    
  (q/no-stroke)
  (q/fill 90 200 200)
  
  ;; the snake
  (doseq [[x y] (:snake state)]
    (q/rect (* x segment-width) (* y segment-width) segment-width segment-width ))

  ;; the apple
  (q/fill 10 200 255)
  (let [[x y] (:apple state)] (q/rect (* x segment-width) (* y segment-width) segment-width segment-width))

  (q/no-fill)
  (q/stroke 230)
  (q/rect 0 0 (+ board-pixels segment-width) (+ board-pixels segment-width))
  (q/no-stroke)

  (if (> (count (:snake state)) 12)
      (q/save-frame "./animation/frame-######.jpg"))
)
  
  


(q/defsketch snake-ai
  :title "Snake AI"
  :size [800 500]
  :setup setup
  :update update-state
  :draw draw-state
  :features [:keep-on-top]
  :middleware [m/fun-mode])
