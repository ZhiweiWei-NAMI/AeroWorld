#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "AeroSemanticTypes.h"
#include "AeroVisualStateReceiver.generated.h"

UINTERFACE(BlueprintType)
class AEROSEMANTICRUNTIME_API UAeroVisualStateReceiver : public UInterface
{
	GENERATED_BODY()
};

class AEROSEMANTICRUNTIME_API IAeroVisualStateReceiver
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Aero|Visual")
	void ApplyAeroVisualState(const FAeroVisualState& VisualState);
};
