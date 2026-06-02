(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    hammer - item
    left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at hammer table)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (goal-at hammer right_storage)
    (handempty)
    (safe hammer)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target hammer)
  )

  (:goal
    (and
      (at hammer right_storage)
    )
  )
)
