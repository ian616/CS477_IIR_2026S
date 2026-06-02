(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    banana_0 - item
    left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at banana_0 table)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (clear banana_0)
    (goal-at banana_0 bookshelf)
    (graspable banana_0)
    (handempty)
    (safe banana_0)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target banana_0)
  )

  (:goal
    (and
      (at banana_0 bookshelf)
    )
  )
)
