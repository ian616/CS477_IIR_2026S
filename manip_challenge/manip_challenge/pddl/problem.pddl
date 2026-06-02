(define (problem manip-generated)
  (:domain manip-tamp)

  (:objects
<<<<<<< Updated upstream
    strawberry_0 strawberry_1 - item
=======
    banana_0 banana_1 coke_can_0 hammer_0 meat_can_0 strawberry_0 strawberry_1 - item
>>>>>>> Stashed changes
    left_storage right_storage bookshelf buffer1 buffer2 - location
  )

  (:init
<<<<<<< Updated upstream
    (at strawberry_0 table)
    (at strawberry_1 table)
=======
    (at banana_0 table)
    (at banana_1 table)
    (at coke_can_0 table)
    (at hammer_0 table)
    (at meat_can_0 table)
    (at strawberry_0 table)
    (at strawberry_1 table)
    (blocks strawberry_1 meat_can_0)
>>>>>>> Stashed changes
    (buffer buffer1)
    (buffer buffer2)
    (buffer-free buffer1)
    (buffer-free buffer2)
<<<<<<< Updated upstream
    (clear strawberry_0)
    (clear strawberry_1)
    (goal-at strawberry_0 right_storage)
    (graspable strawberry_0)
    (graspable strawberry_1)
    (handempty)
=======
    (clear banana_0)
    (clear banana_1)
    (clear coke_can_0)
    (clear hammer_0)
    (clear strawberry_0)
    (clear strawberry_1)
    (goal-at meat_can_0 left_storage)
    (graspable banana_0)
    (graspable banana_1)
    (graspable coke_can_0)
    (graspable hammer_0)
    (graspable meat_can_0)
    (graspable strawberry_0)
    (graspable strawberry_1)
    (handempty)
    (near meat_can_0 strawberry_1)
    (near strawberry_1 meat_can_0)
    (obstacle banana_0)
    (obstacle banana_1)
    (obstacle coke_can_0)
    (obstacle hammer_0)
    (obstacle strawberry_0)
    (obstacle strawberry_1)
    (safe banana_0)
    (safe banana_1)
    (safe coke_can_0)
    (safe hammer_0)
>>>>>>> Stashed changes
    (safe strawberry_0)
    (safe strawberry_1)
    (storage bookshelf)
    (storage left_storage)
    (storage right_storage)
<<<<<<< Updated upstream
    (target strawberry_0)
    (target strawberry_1)
=======
    (target meat_can_0)
>>>>>>> Stashed changes
  )

  (:goal
    (and
<<<<<<< Updated upstream
      (at strawberry_0 right_storage)
=======
      (at meat_can_0 left_storage)
>>>>>>> Stashed changes
    )
  )
)
