(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    coke_can hammer meat_can strawberry_0 - item
    left_storage right_storage bookshelf dynamic_buffer - location
  )

  (:init
    (at strawberry_0 table)
    (buffer dynamic_buffer)
    (buffer-free dynamic_buffer)
    (clear strawberry_0)
    (goal-at strawberry_0 right_storage)
    (graspable strawberry_0)
    (handempty)
    (obstacle strawberry_0)
    (safe strawberry_0)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
  )

  (:goal
    (and
      (at strawberry_0 right_storage)
    )
  )
)
