// Copyright (C) Alibaba.inc. DataV, 2024. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "CommonStructsDefine.h"
#include "DtfGeomStructsDefine.h"
#include "Math/MathFwd.h"
#include "ArchitectureType.generated.h"

USTRUCT(BlueprintType)
struct FDynamicBuildingTypeGroup
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "TypeGroup"), Category = "Default")
	TArray<FName> TypeGroup;
};

USTRUCT(BlueprintType)
struct FDynamicBuildingStyle_01
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MainStructure", MakeStructureDefaultValue = "(CornerWidth_9_D9409EA6486764E95BF0EB9A8AF39D90=200.000000,TopStructureHeight_12_EB6FACE94283725BFFB2E285AA652E74=(X=600.000000,Y=200.000000),Offsets_15_BC55D43F4B0FF8CDC1D172B7F58BBEF6=(X=-20.000000,Y=100.000000),HasTopStructure_17_AECA21804BB044FFC74273B8008A2039=False)"), Category = "Default")
	FDynamicBuildingMainStructStyle_01 MainStructure;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Default", meta = (DisplayName = "FinOn", MakeStructureDefaultValue = "True"))
	bool FinOn;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Fin", MakeStructureDefaultValue = "(HorizontalRhythm_32_E327810D434EA22B1B896BA3E82AA084=(X=2.000000,Y=4.000000),MainFin_24_8CED13CB43BA06C730AD749A5577A6E3=(Offsets_22_8CED13CB43BA06C730AD749A5577A6E3=(X=-20.000000,Y=100.000000),Thickness_23_535DE7144157DF2516626B9A93B32804=50.000000),SubFin_23_4DB4FBE64B145E076254B2A90C030D1C=(Offsets_22_8CED13CB43BA06C730AD749A5577A6E3=(X=-50.000000,Y=30.000000),Thickness_23_535DE7144157DF2516626B9A93B32804=10.000000))"), Category = "Default")
	FDynamicBuildingFinGroupStyle Fin;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Default", meta = (DisplayName = "LouverOn", MakeStructureDefaultValue = "True"))
	bool LouverOn;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Louver", MakeStructureDefaultValue = "(VerticalRhythm_31_E327810D434EA22B1B896BA3E82AA084=(X=0.000000,Y=0.000000),MainLouver_28_8CED13CB43BA06C730AD749A5577A6E3=(Offsets_22_8CED13CB43BA06C730AD749A5577A6E3=(X=0.000000,Y=0.000000),Thickness_23_535DE7144157DF2516626B9A93B32804=0.000000),SubLouver_29_4DB4FBE64B145E076254B2A90C030D1C=(Offsets_22_8CED13CB43BA06C730AD749A5577A6E3=(X=0.000000,Y=0.000000),Thickness_23_535DE7144157DF2516626B9A93B32804=0.000000))"), Category = "Default")
	FDynamicBuildingLouverGroupStyle Louver;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Parapet"), Category = "Default")
	FDynamicBuildingParapet Parapet;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Facade"), Category = "Default")
	FDynamicBuildingFacadeStyle Facade;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "TopFacadeMat"), Category = "Default")
	FDynamicBuildingFacadeLayout TopFacadeMat;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MainFacadeMat"), Category = "Default")
	FDynamicBuildingFacadeLayout MainFacadeMat;

	FDynamicBuildingStyle_01()
		: MainStructure(200.0, FVector2D(600.0, 200.0), FVector2D(-20.0, 100.0), false)
		, FinOn(true)
		, Fin()
		, LouverOn(true)
		, Louver(FVector2D(0.0, 0.0), FDynamicBuildingSinglePanelStyle(FVector2D(0.0, 0.0), 0.0), FDynamicBuildingSinglePanelStyle(FVector2D(0.0, 0.0), 0.0))
		, Parapet()
		, Facade(450.0, 450.0, 450.0)
		, TopFacadeMat(nullptr, FVector2D(6.0, 6.0))
		, MainFacadeMat(nullptr, FVector2D(6.0, 6.0)) {}
};


USTRUCT(BlueprintType)
struct FDynamicBuildingStyle_02
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "FinOn", MakeStructureDefaultValue = "False"), Category = "Default")
	bool FinOn;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "FinStyle"), Category = "Default")
	FDynamicBuildingFinGroupStyle FinStyle;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "LouverOn", MakeStructureDefaultValue = "False"), Category = "Default")
	bool LouverOn;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "LouverStyle"), Category = "Default")
	FDynamicBuildingLouverGroupStyle LouverStyle;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Parapet"), Category = "Default")
	FDynamicBuildingParapet Parapet;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MainFacade"), Category = "Default")
	FDynamicBuildingFacadeLayout MainFacade;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "BoxFacade"), Category = "Default")
	FDynamicBuildingFacadeLayout BoxFacade;

	FDynamicBuildingStyle_02()
		: FinOn(false)
		, FinStyle()
		, LouverOn(false)
		, LouverStyle()
		, Parapet()
		, MainFacade(nullptr, FVector2D(6.0, 4.0))
		, BoxFacade(nullptr, FVector2D(1.0, 1.0)) {}

};

USTRUCT(BlueprintType)
struct FDynamicBuildingStyle_03
{
	GENERATED_BODY()
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MaterialList", MakeStructureDefaultValue = "()"), Category = "Default")
	TMap<EDynamicBuildingResidenceElement ,FDynamicBuildingMaterialGroup> MaterialList;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Color", MakeStructureDefaultValue = "(R=0.000000,G=0.000000,B=0.000000,A=0.000000)"), Category = "Default")
	FLinearColor Color;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Rate", MakeStructureDefaultValue = "0.200000"), Category = "Default")
	double Rate;
};


/** Please add a struct description */
USTRUCT(BlueprintType)
struct FFStruct_IntValueForLanduse
{
	GENERATED_BODY()
public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "商业", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 commercial;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "工地", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 construction;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "教育", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 education;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "会展", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 fairground;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "工业", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 industrial;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "住宅", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 residential;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "零售", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 retail;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "其他", MakeStructureDefaultValue = "0"), Category = "Default")
	int32 other;
};

/** Please add a struct description */
USTRUCT(BlueprintType)
struct FFStruct_FloatValueForLanduse
{
	GENERATED_BODY()
public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "商业", MakeStructureDefaultValue = "0"), Category = "Default")
	double commercial;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "工地", MakeStructureDefaultValue = "0"), Category = "Default")
	double construction;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "教育", MakeStructureDefaultValue = "0"), Category = "Default")
	double education;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "会展", MakeStructureDefaultValue = "0"), Category = "Default")
	double fairground;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "工业", MakeStructureDefaultValue = "0"), Category = "Default")
	double industrial;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "住宅", MakeStructureDefaultValue = "0"), Category = "Default")
	double residential;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "零售", MakeStructureDefaultValue = "0"), Category = "Default")
	double retail;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "其他", MakeStructureDefaultValue = "0"), Category = "Default")
	double other;
};

/** Please add a struct description */
USTRUCT(BlueprintType)
struct FArchitectureColorGroup
{
	GENERATED_BODY()
public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Main", MakeStructureDefaultValue = "(R=1.000000,G=1.000000,B=1.000000,A=1.000000)"), Category = "Default")
	FLinearColor Main;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Sub", MakeStructureDefaultValue = "(R=1.000000,G=1.000000,B=1.000000,A=1.000000)"), Category = "Default")
	FLinearColor Sub;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Facade", MakeStructureDefaultValue = "(R=1.000000,G=1.000000,B=1.000000,A=1.000000)"), Category = "Default")
	FLinearColor Facade;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Glass", MakeStructureDefaultValue = "(R=1.000000,G=1.000000,B=1.000000,A=1.000000)"), Category = "Default")
	FLinearColor Glass;
};

/** Please add a struct description */
USTRUCT(BlueprintType)
struct FArchiParentInfo
{
	GENERATED_BODY()
public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "LevelNum", MakeStructureDefaultValue = "20"), Category = "Default")
	int32 LevelNum;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "LevelHeight", MakeStructureDefaultValue = "420.000000"), Category = "Default")
	double LevelHeight;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "TopLevel", MakeStructureDefaultValue = "2"), Category = "Default")
	int32 TopLevel;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "BottomLevel", MakeStructureDefaultValue = "2"), Category = "Default")
	int32 BottomLevel;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "UseData", MakeStructureDefaultValue = "True"), Category = "Default")
	bool UseData;

		/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "HasMainPitchedRoof", MakeStructureDefaultValue = "False"), Category = "Default")
	bool HasMainPitchedRoof;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Landuse", MakeStructureDefaultValue = "Residential"), Category = "Default")
	EDynamicBuildingLanduse Landuse;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "MyArea", MakeStructureDefaultValue = "100.000000"), Category = "Default")
	double MyArea;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Tag"), Category = "Default")
	FString Tag;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Style", MakeStructureDefaultValue = "S1"), Category = "Default")
	FName Style;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "ColorGroup", MakeStructureDefaultValue = "(Main=(R=1.000000,G=1.000000,B=1.000000,A=1.000000),Sub=(R=1.000000,G=1.000000,B=1.000000,A=1.000000),Facade=(R=1.000000,G=1.000000,B=1.000000,A=1.000000),Glass=(R=1.000000,G=1.000000,B=1.000000,A=1.000000))"), Category = "Default")
	FArchitectureColorGroup ColorGroup;
};

/** Please add a struct description */
USTRUCT(BlueprintType)
struct FArchiGeneratingInfo
{
	GENERATED_BODY()
public:
	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Outline"), Category = "Default")
	FDtfGeomPolyline Outline;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Class"), Category = "Default")
	TSubclassOf<class ADynamicMeshActor> Class;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Info"), Category = "Default")
	FArchiParentInfo Info;

	/** Please add a variable description */
	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (DisplayName = "Transform"), Category = "Default")
	FTransform Transform;
};
