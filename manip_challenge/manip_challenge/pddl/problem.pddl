(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
    banana_0 coke_can_0 hammer_0 meat_can_0 strawberry - item
    left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
    (at banana_0 table)
    (at coke_can_0 table)
    (at hammer_0 table)
    (at meat_can_0 table)
    (at strawberry table)
    (blocks meat_can_0 coke_can_0)
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
    (clear banana_0)
    (clear hammer_0)
    (clear meat_can_0)
    (goal-at banana_0 bookshelf)
    (graspable banana_0)
    (graspable coke_can_0)
    (graspable hammer_0)
    (graspable meat_can_0)
    (handempty)
    (obstacle coke_can_0)
    (obstacle hammer_0)
    (obstacle meat_can_0)
    (obstacle strawberry)
    (safe banana_0)
    (safe coke_can_0)
    (safe hammer_0)
    (safe meat_can_0)
    (safe strawberry)
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
