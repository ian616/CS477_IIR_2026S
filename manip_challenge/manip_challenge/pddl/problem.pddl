(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    banana_0 meat_can_0 strawberry_0 - item
    left_storage right_storage bookshelf dynamic_buffer - location
  )

  (:init
    (at banana_0 table)
    (at meat_can_0 table)
    (at strawberry_0 table)
    (buffer dynamic_buffer)
    (buffer-free dynamic_buffer)
    (clear banana_0)
    (clear meat_can_0)
    (clear strawberry_0)
    (goal-at strawberry_0 bookshelf)
    (graspable banana_0)
    (graspable meat_can_0)
    (graspable strawberry_0)
    (handempty)
    (obstacle banana_0)
    (obstacle meat_can_0)
    (safe banana_0)
    (safe meat_can_0)
    (safe strawberry_0)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
    (target strawberry_0)
  )

  (:goal
    (and
      (at strawberry_0 bookshelf)
    )
  )
)
