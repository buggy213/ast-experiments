(define (recursive_fact n) (if (<= n 1) 1 (* n (recursive-fact (- n 1)))))
(define (tail_fact n)
    (define (fact_iter x y) 
        (if (> x n) 
            y
            (fact_iter (+ x 1) (* x y))
        )
    )
    (fact_iter 1 1)
)

(recursive_fact 5)
(tail_fact 5)