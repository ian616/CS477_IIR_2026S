(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    banana coke_can hammer_0 meat_can strawberry - item
    left_storage right_storage bookshelf dynamic_buffer - location
  )

  (:init
    (buffer dynamic_buffer)
    (buffer-free dynamic_buffer)
    (goal-at banana left_storage)
    (handempty)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
  )

  (:goal
    (and
      (at banana left_storage)
    )
  )
)
