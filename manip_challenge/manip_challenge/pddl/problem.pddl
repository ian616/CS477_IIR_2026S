(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    strawberry_0 - item
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
    (safe strawberry_0)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target strawberry_0)
  )

  (:goal
    (and
      (at strawberry_0 right_storage)
    )
  )
)
