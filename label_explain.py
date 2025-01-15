import json

SemEval_label2exp = {
    "CAUSE AND EFFECT":"entity 1 causes entity 2 to occur (e.g., 'pollution' causes 'disease').", 
    "COMPONENT AND WHOLE":"Entity 1 is a part of Entity 2 (e.g., 'tire' is a part of 'car').", 
    "ENTITY AND DESTINATION":"Entity 1's destination is Entity 2 (e.g., 'immigrant' moves to 'United States').",
    "ENTITY AND ORIGIN":"Entity 1 originates from Entity 2 (e.g., 'person' comes from 'Europe').",
    "PRODUCT AND PRODUCER":"Entity 1 is produced by Entity 2 (e.g., 'cake' is made by 'baking').",
    "MEMBER AND COLLECTION":"Entity 1 is a member of Entity 2 (e.g., 'policeman' is a member of 'police station').",
    "MESSAGE AND TOPIC":"Entity 1 is information about Entity 2 (e.g., 'speech' is about 'economy').",
    "CONTENT AND CONTAINER":"Entity 1 is in Entity 2 (e.g., 'apple' is in 'basket').",
    "INSTRUMENT AND AGENCY":"Entity 1 is a tool or method of Entity 2 (e.g., 'tool' is used for 'cutting').",
    "NONE":"If the relationship does not belong to any of the above, it is classified as 'NONE'."
}